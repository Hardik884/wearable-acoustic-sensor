# Architecture

## SPINE body-sensor-network layout

```
 forearm                                  laptop
┌─────────────────────────────┐          ┌──────────────────────────────────────────────────┐
│ NODE (ESP32 / ESP32-S3)     │  USB     │ COORDINATOR (Python)                             │
│                             │  serial  │                                                  │
│ mic_0 ─┐                    │ 921600   │  Transport  ◄── SERIAL | SIMULATED (one flag)    │
│ mic_1 ─┤ ADC1 @10kHz ring   │ ───────► │     │                                            │
│        │ buffer + trigger   │ DATA     │  FrameParser (CRC, resync)                       │
│ act_0 ◄┘ chirp probe        │ LIVE     │     │                                            │
│                             │ ADVERT   │  Function pipeline (SPINE function library)      │
│ SPINE node service          │ ◄─────── │   F_PREPROCESS → F_FEATURES → F_LOCALIZE         │
│ (discovery, setup, start)   │ SETUP... │                    → F_CLASSIFY → F_FUSION       │
└─────────────────────────────┘          │     │                                            │
                                         │  Listener API: on_new_node / on_data /           │
                                         │                on_live / on_service_message      │
                                         │     ├─ Dashboard app (FastAPI + WebSocket → UI)  │
                                         │     ├─ Safety app    (rules → alerts, SQLite)    │
                                         │     ├─ Calibration app (profiles/<user>.json)    │
                                         │     └─ Recorder app  (data/real/<m>/<z>/<n>.npz) │
                                         └──────────────────────────────────────────────────┘
```

## How SPINE is used

| SPINE concept | Here |
|---|---|
| Node | ESP32 firmware: sensing, tap triggering, probe generation, framing |
| Coordinator | `spine/coordinator.py`: discovers the node, configures it, starts it, receives data, runs processing, hosts apps |
| Service discovery / advertisement | `SERVICE_DISCOVERY` → `SERVICE_ADVERTISEMENT` (sensors, actuator, rate, firmware, current config); shown on the SPINE NODE tab |
| Setup commands | `SETUP_SENSOR`, `SETUP_PROBE`, `CALIBRATION_UPDATE`, `START`, `STOP`, `RESET` |
| Functions | `spine/functions/library.py`: registry of `F_*` blocks (code + parameters); the pipeline is a list of `{code, params}` |
| Applications | listeners that never touch the transport; they cannot tell SERIAL from SIMULATED |
| Protocol | `protocol/spine_protocol.md`, implemented once in Python and once in C++, identical bytes proven by golden vectors |

Data flow for one tap: node trigger → capture pre + tap window → chirp probe → capture response → one
`DATA` frame → coordinator → `F_PREPROCESS` (DC removal, gain correction, onset, baseline-normalised probe
bands) → `F_FEATURES` (RMS, peak, centroid, rolloff, MFCC×8, attack, decay, level ratio, GCC-PHAT TDOA,
probe band energies) → `F_LOCALIZE` (zone model) → `F_CLASSIFY` (material model) → `F_FUSION` (unknown
thresholding, overall confidence) → listeners.

## Transports

* `SerialTransport` – pyserial, 921600 baud, port auto-detect by sending `SERVICE_DISCOVERY` to each port.
* `SimTransport` – `VirtualNode` (spine/sim) generates physics-inspired SYNTHETIC taps and speaks the same
  frames. Chosen by `python run.py demo --sim`.

## Next phase

1. **Move SPINE functions onto the node.** `F_PREPROCESS` + a reduced `F_FEATURES` (RMS, peak, centroid,
   decay, level ratio, TDOA) fit comfortably on an ESP32-S3; quantise the RandomForest / a small MLP and
   run it with TFLite-Micro / emlearn. The node then sends a ~16-byte classification message instead of
   7 KB of waveform, with raw DATA only on request (SPINE "function on node" model — `SETUP_FUNCTION`
   message would carry the function code + parameters).
2. **Switch USB → BLE.** Only a new `BleTransport` (NUS or custom GATT) is needed; the on-node
   classification message fits one notification, `LIVE` can be reduced to a coarse envelope.
3. Per-user profile upload to the node (threshold/gain) already works via `CALIBRATION_UPDATE`; extend
   it with a gain value and a stored model id.
4. More zones / contact types: add classes in `config.yaml`, record, retrain — no code changes.
