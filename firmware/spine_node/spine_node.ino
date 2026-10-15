// SPINE node firmware: wearable acoustic sensor (ESP32 / ESP32-S3).
//
//   mic_0 (wrist) + mic_1 (5 cm toward the elbow): piezo discs, 1 MOhm/1 MOhm mid-rail bias
//   act_0: piezo disc driven by two GPIOs in opposite phase
//   Protocol: see protocol/spine_protocol.md (spine_protocol.h is the C++ implementation)
//
// Build: Arduino IDE (board "ESP32 Dev Module" or "ESP32S3 Dev Module", open this folder's
// .ino) or PlatformIO (firmware/platformio.ini).  Serial baud: 921600.
#include <Arduino.h>
#include "spine_protocol.h"
#include "soc/gpio_reg.h"
#include "soc/soc.h"

// ------------------------------------------------------------------ board select
// 1 = ESP32-S3, 0 = classic ESP32. Auto-detected from the selected board; force it here if needed.
#ifndef BOARD_IS_S3
  #if defined(CONFIG_IDF_TARGET_ESP32S3)
    #define BOARD_IS_S3 1
  #else
    #define BOARD_IS_S3 0
  #endif
#endif

#if BOARD_IS_S3
  // ADC1 = GPIO1..10 on the S3. Actuator pins must be < 32 (fast register writes below).
  #define PIN_MIC0   1
  #define PIN_MIC1   2
  #define PIN_ACT_A  15
  #define PIN_ACT_B  16
#else
  // ADC1 = GPIO32..39 on the ESP32 (34/35 are input-only, perfect for the mics).
  #define PIN_MIC0   34
  #define PIN_MIC1   35
  #define PIN_ACT_A  25
  #define PIN_ACT_B  26
#endif

// ------------------------------------------------------------------ constants
static const uint16_t FW_VERSION = 0x0100;       // 1.0
static const uint32_t SERIAL_BAUD = 921600;
static const uint8_t  NODE_ID = 1;
static const uint32_t TICK_HZ = 20000;           // timer tick: actuator waveform + ADC pacing
static const uint16_t DEFAULT_RATE = 10000;      // per-channel ADC rate (5000/10000/20000)
static const uint16_t DEFAULT_THRESHOLD = 120;   // counts of deviation from mid-rail
static const uint16_t DEFAULT_REFRACTORY_MS = 300;
static const uint16_t PRE_MS = 20, TAP_MS = 100;
static const uint16_t SKIP_MS = 3, TAIL_MS = 10; // probe: drop direct coupling, keep ring-down
static const uint16_t LIVE_RATE = 1000;          // LIVE waveform rate (peak-preserving decimation)
static const uint8_t  LIVE_N = 50;               // samples per LIVE frame (50 ms)

// worst case buffers (20 kHz): pre 400 + tap 2000, probe 1200
static const int RING = 4096;                    // samples per channel (power of 2)
static const int MAX_PT = 2400, MAX_PROBE = 1200;

// ------------------------------------------------------------------ shared state
static volatile uint16_t ring[2][RING];
static volatile uint32_t head = 0;               // total samples written (per channel)

static int16_t evt[2][MAX_PT];                   // pre-trigger + tap window (DC removed)
static int16_t prb[2][MAX_PROBE];                // probe response window
static volatile bool evt_ready = false;
static uint16_t evt_n_pre = 0, evt_n_tap = 0, evt_n_probe = 0, evt_trig_level = 0;
static uint8_t evt_flags = 0;
static uint32_t evt_ts = 0;

static int16_t live_buf[2][LIVE_N];
static volatile bool live_ready = false;

static volatile bool running = false, live_on = false;
static volatile uint16_t rate_hz = DEFAULT_RATE, threshold = DEFAULT_THRESHOLD;
static volatile uint16_t refractory_ms = DEFAULT_REFRACTORY_MS;
static volatile uint32_t overruns = 0;

static struct { bool enabled = true; uint16_t f0 = 300, f1 = 4000, dur_ms = 50; } probe;

// ------------------------------------------------------------------ actuator (ISR side)
static volatile bool drive_on = false;
static volatile int32_t drive_ticks_left = 0;
static uint32_t drive_phase = 0;
static int64_t drive_inc_q16 = 0, drive_dinc_q16 = 0;   // phase increment per tick, Q16

static TaskHandle_t sample_task = nullptr;
static volatile uint32_t sample_div = TICK_HZ / DEFAULT_RATE;
static uint32_t tick_count = 0;

static inline void IRAM_ATTR act_set(uint32_t on_mask, uint32_t off_mask) {
  REG_WRITE(GPIO_OUT_W1TS_REG, on_mask);
  REG_WRITE(GPIO_OUT_W1TC_REG, off_mask);
}
static const uint32_t MASK_A = 1UL << PIN_ACT_A, MASK_B = 1UL << PIN_ACT_B;

void IRAM_ATTR onTick() {
  if (drive_on) {
    if (drive_ticks_left-- <= 0) {
      drive_on = false;
      act_set(0, MASK_A | MASK_B);
    } else {
      drive_phase += (uint32_t)(drive_inc_q16 >> 16);
      drive_inc_q16 += drive_dinc_q16;
      if (drive_phase & 0x80000000UL) {      // square wave, opposite phase on the two pins
        act_set(MASK_A, MASK_B);
      } else {
        act_set(MASK_B, MASK_A);
      }
    }
  }
  if (++tick_count >= sample_div) {
    tick_count = 0;
    BaseType_t woken = pdFALSE;
    vTaskNotifyGiveFromISR(sample_task, &woken);
    if (woken) portYIELD_FROM_ISR();
  }
}

static void start_chirp(uint16_t f0, uint16_t f1, uint16_t dur_ms) {
  uint32_t n = (uint32_t)TICK_HZ * dur_ms / 1000;
  int64_t inc0 = ((int64_t)f0 << 32) / TICK_HZ, inc1 = ((int64_t)f1 << 32) / TICK_HZ;
  drive_phase = 0;
  drive_inc_q16 = inc0 << 16;
  drive_dinc_q16 = ((inc1 - inc0) << 16) / (int64_t)n;
  drive_ticks_left = n;
  drive_on = true;
}

// ------------------------------------------------------------------ sampling task
enum { ST_ARMED, ST_TAP, ST_PROBE_SKIP, ST_PROBE_REC, ST_DONE };

static void sampling_task(void*) {
  float dc[2] = {2048, 2048};
  uint8_t st = ST_ARMED;
  uint32_t trig_idx = 0, probe_idx = 0, wait = 0, refractory_until = 0;
  uint16_t n_pre = 0, n_tap = 0, n_skip = 0, n_probe = 0;
  uint16_t dc_snap[2] = {2048, 2048};
  uint8_t dec = 10, dec_i = 0;
  int16_t pk[2] = {0, 0};
  uint8_t live_i = 0;
  uint16_t cur_rate = 0;

  for (;;) {
    uint32_t n = ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    if (n > 1) overruns += n - 1;

    uint16_t r = rate_hz;
    if (r != cur_rate) {                          // (re)configure windows after SETUP_SENSOR
      cur_rate = r;
      n_pre = (uint32_t)r * PRE_MS / 1000;
      n_tap = (uint32_t)r * TAP_MS / 1000;
      n_skip = (uint32_t)r * SKIP_MS / 1000;
      dec = r / LIVE_RATE;
      st = ST_ARMED; dec_i = 0; live_i = 0;
    }

    uint16_t raw0 = analogRead(PIN_MIC0), raw1 = analogRead(PIN_MIC1);
    uint32_t w = head & (RING - 1);
    ring[0][w] = raw0; ring[1][w] = raw1;
    head = head + 1;
    float d0 = raw0 - dc[0], d1 = raw1 - dc[1];
    float a = fabsf(d0) > fabsf(d1) ? fabsf(d0) : fabsf(d1);

    // slow DC (mid-rail) tracking, only on quiet samples
    if (a < threshold) { dc[0] += d0 * (1.0f / 256); dc[1] += d1 * (1.0f / 256); }

    // LIVE: peak-preserving decimation to 1 kHz
    if (live_on && running) {
      int16_t v0 = (int16_t)d0, v1 = (int16_t)d1;
      if (dec_i == 0) { pk[0] = v0; pk[1] = v1; }
      else {
        if (abs(v0) > abs(pk[0])) pk[0] = v0;
        if (abs(v1) > abs(pk[1])) pk[1] = v1;
      }
      if (++dec_i >= dec) {
        dec_i = 0;
        live_buf[0][live_i] = pk[0]; live_buf[1][live_i] = pk[1];
        if (++live_i >= LIVE_N) { live_i = 0; if (!live_ready) live_ready = true; }
      }
    }

    uint32_t now = millis();
    switch (st) {
      case ST_ARMED:
        if (running && !evt_ready && (int32_t)(now - refractory_until) >= 0 && a > threshold) {
          trig_idx = head - 1;
          dc_snap[0] = (uint16_t)dc[0]; dc_snap[1] = (uint16_t)dc[1];
          evt_trig_level = (uint16_t)a;
          evt_ts = now;
          refractory_until = now + refractory_ms;
          wait = n_tap - 1;
          st = ST_TAP;
        }
        break;
      case ST_TAP:
        if (wait-- == 0) {
          uint32_t start = trig_idx - n_pre;       // copy [trig-pre, trig+tap) from the ring
          for (uint16_t i = 0; i < n_pre + n_tap; i++) {
            uint32_t k = (start + i) & (RING - 1);
            evt[0][i] = (int16_t)((int)ring[0][k] - (int)dc_snap[0]);
            evt[1][i] = (int16_t)((int)ring[1][k] - (int)dc_snap[1]);
          }
          evt_n_pre = n_pre; evt_n_tap = n_tap; evt_flags = 0; evt_n_probe = 0;
          if (probe.enabled) {
            n_probe = (uint32_t)cur_rate * (probe.dur_ms + TAIL_MS - SKIP_MS) / 1000;
            if (n_probe > MAX_PROBE) n_probe = MAX_PROBE;
            start_chirp(probe.f0, probe.f1, probe.dur_ms);
            wait = n_skip; probe_idx = 0;
            st = ST_PROBE_SKIP;
          } else {
            evt_ready = true;
            st = ST_ARMED;
          }
        }
        break;
      case ST_PROBE_SKIP:                         // ignore direct drive coupling
        if (wait-- == 0) st = ST_PROBE_REC;
        break;
      case ST_PROBE_REC:
        prb[0][probe_idx] = (int16_t)((int)raw0 - (int)dc_snap[0]);
        prb[1][probe_idx] = (int16_t)((int)raw1 - (int)dc_snap[1]);
        if (++probe_idx >= n_probe) {
          evt_n_probe = n_probe; evt_flags = spine::FLAG_PROBE_USED;
          refractory_until = millis() + 50;       // let the ring-down die out
          evt_ready = true;
          st = ST_ARMED;
        }
        break;
      default: st = ST_ARMED;
    }
  }
}

// ------------------------------------------------------------------ serial output
struct BufSink {                                  // batches byte-wise writes into 256-byte chunks
  uint8_t buf[256]; size_t n = 0;
  void write(const uint8_t* p, size_t len) {
    for (size_t i = 0; i < len; i++) { buf[n++] = p[i]; if (n == sizeof(buf)) flush(); }
  }
  void flush() { if (n) { Serial.write(buf, n); n = 0; } }
};

static uint16_t tx_seq = 0;

static void send_message(uint8_t code, uint8_t ref, const char* text) {
  BufSink s; spine::FrameWriter<BufSink> w(s);
  size_t len = strlen(text);
  w.begin(spine::SERVICE_MESSAGE, NODE_ID, ++tx_seq, 2 + len);
  w.u8(code); w.u8(ref); w.bytes((const uint8_t*)text, len);
  w.end(); s.flush();
}

static void send_advert() {
  BufSink s; spine::FrameWriter<BufSink> w(s);
  w.begin(spine::SERVICE_ADVERTISEMENT, NODE_ID, ++tx_seq, 19);
  w.u8(2); w.u8(1); w.u16(rate_hz); w.u16(FW_VERSION); w.u8(12); w.u8(BOARD_IS_S3);
  w.u16(threshold); w.u16(refractory_ms); w.u8(probe.enabled); w.u16(probe.f0); w.u16(probe.f1);
  w.u16(probe.dur_ms);
  w.end(); s.flush();
}

static void send_event() {
  BufSink s; spine::FrameWriter<BufSink> w(s);
  uint16_t n_pt = evt_n_pre + evt_n_tap;
  w.begin(spine::DATA, NODE_ID, ++tx_seq, 16 + 4 * n_pt + 4 * evt_n_probe);
  w.u32(evt_ts); w.u16(rate_hz); w.u8(2); w.u8(evt_flags); w.u16(evt_n_pre); w.u16(evt_n_tap);
  w.u16(evt_n_probe); w.u16(evt_trig_level);
  for (int c = 0; c < 2; c++) for (uint16_t i = 0; i < n_pt; i++) w.i16(evt[c][i]);
  for (int c = 0; c < 2; c++) for (uint16_t i = 0; i < evt_n_probe; i++) w.i16(prb[c][i]);
  w.end(); s.flush();
}

static void send_live() {
  BufSink s; spine::FrameWriter<BufSink> w(s);
  w.begin(spine::LIVE, NODE_ID, ++tx_seq, 8 + 4 * LIVE_N);
  w.u32(millis()); w.u16(LIVE_RATE); w.u8(2); w.u8(LIVE_N);
  for (int c = 0; c < 2; c++) for (uint8_t i = 0; i < LIVE_N; i++) w.i16(live_buf[c][i]);
  w.end(); s.flush();
}

// ------------------------------------------------------------------ command handling
static spine::FrameParser parser;
static spine::Frame frame;

static void defaults() {
  rate_hz = DEFAULT_RATE; sample_div = TICK_HZ / DEFAULT_RATE;
  threshold = DEFAULT_THRESHOLD; refractory_ms = DEFAULT_REFRACTORY_MS;
  probe.enabled = true; probe.f0 = 300; probe.f1 = 4000; probe.dur_ms = 50;
}

static void handle(const spine::Frame& f) {
  using namespace spine;
  switch (f.type) {
    case SERVICE_DISCOVERY: send_advert(); break;
    case SETUP_SENSOR: {
      if (f.len < 6) { send_message(CODE_ERROR, f.type, "short"); break; }
      uint16_t r = rd16(f.payload);
      if (r != 5000 && r != 10000 && r != 20000) { send_message(CODE_ERROR, f.type, "bad rate"); break; }
      sample_div = TICK_HZ / r;
      rate_hz = r;
      threshold = rd16(f.payload + 2);
      refractory_ms = rd16(f.payload + 4);
      send_message(CODE_ACK, f.type, "ok");
      break;
    }
    case SETUP_PROBE: {
      if (f.len < 7) { send_message(CODE_ERROR, f.type, "short"); break; }
      uint16_t f0 = rd16(f.payload + 1), f1 = rd16(f.payload + 3), d = rd16(f.payload + 5);
      if (d < 5 || d > 100 || f0 < 20 || f1 < 20 || f0 > rate_hz / 2 || f1 > rate_hz / 2) {
        send_message(CODE_ERROR, f.type, "bad probe"); break;
      }
      probe.enabled = f.payload[0] != 0; probe.f0 = f0; probe.f1 = f1; probe.dur_ms = d;
      send_message(CODE_ACK, f.type, "ok");
      break;
    }
    case CALIBRATION_UPDATE:
      if (f.len < 4) { send_message(CODE_ERROR, f.type, "short"); break; }
      threshold = rd16(f.payload);
      refractory_ms = rd16(f.payload + 2);
      send_message(CODE_ACK, f.type, "ok");
      break;
    case START:
      live_on = f.len > 0 && f.payload[0] != 0;
      running = true;
      send_message(CODE_ACK, f.type, "started");
      break;
    case STOP:
      running = false;
      send_message(CODE_ACK, f.type, "stopped");
      break;
    case RESET:
      running = false; defaults();
      send_message(CODE_ACK, f.type, "reset");
      break;
    default:
      send_message(CODE_ERROR, f.type, "unknown");
  }
}

// ------------------------------------------------------------------ setup / loop
void setup() {
  Serial.begin(SERIAL_BAUD);
  pinMode(PIN_ACT_A, OUTPUT); pinMode(PIN_ACT_B, OUTPUT);
  digitalWrite(PIN_ACT_A, LOW); digitalWrite(PIN_ACT_B, LOW);
  analogReadResolution(12);
  analogSetPinAttenuation(PIN_MIC0, ADC_11db);
  analogSetPinAttenuation(PIN_MIC1, ADC_11db);
  pinMode(PIN_MIC0, INPUT); pinMode(PIN_MIC1, INPUT);

  xTaskCreatePinnedToCore(sampling_task, "sample", 4096, nullptr, configMAX_PRIORITIES - 2,
                          &sample_task, 0);

  hw_timer_t* timer;
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  timer = timerBegin(1000000);                    // 1 MHz base
  timerAttachInterrupt(timer, &onTick);
  timerAlarm(timer, 1000000 / TICK_HZ, true, 0);
#else
  timer = timerBegin(0, 80, true);                // 80 MHz / 80 = 1 MHz
  timerAttachInterrupt(timer, &onTick, true);
  timerAlarmWrite(timer, 1000000 / TICK_HZ, true);
  timerAlarmEnable(timer);
#endif
}

void loop() {
  while (Serial.available()) {
    if (parser.push((uint8_t)Serial.read(), frame)) handle(frame);
  }
  if (evt_ready) {
    send_event();
    evt_ready = false;
  }
  if (live_ready) {
    send_live();
    live_ready = false;
  }
}
