# Wearable Acoustic Sensor — on-body touch + material recognition

A band on the forearm listens to taps through the skin and tells **where** the tap was (zone) and
**what object** made it (material). ESP32 node + Python coordinator, SPINE-style protocol, FastAPI dashboard.

> **Honesty note:** everything labelled SYNTHETIC / SIMULATED comes from a physics-inspired generator, not
> from a real arm. Synthetic accuracy is a software sanity check, **not a result**. Only models trained on
> `data/real/` (source = `real` or `mixed`) can be quoted as real.

## 0. One-time setup (laptop)

```bash
pip install -r requirements.txt
python run.py test          # 13 tests, ~1 min
```

## Backup demo — no hardware needed (do this first, 2 minutes)

```bash
python run.py demo --sim     # or double-click demo_sim.bat
```

Opens http://127.0.0.1:8000. First start trains on SYNTHETIC data if no model exists (~30 s). The
header shows an orange **SIMULATED** badge. Use "Simulate tap" on the LIVE tab, or wait for auto taps.
Try the Safety tab (metal on wrist → red DANGER banner + beep), Calibration, Recorder and Model tabs.

## Tonight, with hardware — in order

### 1. Wire (see `WIRING.md`)
ESP32: mic_0 = GPIO34, mic_1 = GPIO35, actuator = GPIO25 + GPIO26. Two 1 MΩ per mic (3V3 — pin — GND),
piezo between pin and GND.

### 2. Flash
*Arduino IDE:* install "esp32" boards package (2.x or 3.x), open `firmware/spine_node/spine_node.ino`,
board = *ESP32 Dev Module* (or *ESP32S3 Dev Module*), Upload. (ESP32-S3 with native USB: either set
"USB CDC On Boot: Enabled" or use the UART port; baud 921600 is ignored on native USB.)

*PlatformIO:*
```bash
cd firmware && pio run -e esp32 -t upload      # or -e esp32s3
```

### 3. Check the signal (before anything else)
```bash
python run.py ports                  # find the COM port
python run.py monitor --port COM5    # idle: two small bars; tap a mic: it spikes. Ctrl+C to stop
```
If `--port` is omitted the port is auto-detected. Idle values should be ~10–30 counts, taps hundreds.

### 4. Open the dashboard on the real device
```bash
python run.py demo --port COM5       # badge must say REAL DEVICE
```
LIVE tab: waveform moves, taps light the zone. (The first model is trained on synthetic data and will
**not** classify real taps well — that's expected. Next steps fix that.)

### 5. Calibrate (CALIBRATION tab, ~3 min)
Name → *Start new profile* → run steps 1–4 in order (silence → gain → probe baseline → 5 samples per
zone×material). Step 5 *Adapt model* retrains with your samples.

### 6. Record real training data (RECORDER tab) — most valuable step
```bash
python run.py record --port COM5
```
Pick material + zone, target 40, *Start recording*, tap; each tap flashes + counts; *Delete last sample*
undoes a bad one. Fill the progress grid (4 materials × 2 zones × ~40 = 320 taps ≈ 15 min). Files go to
`data/real/<material>/<zone>/<n>.npz`.

### 7. Train
```bash
python run.py train --source mixed      # synthetic pre-training + your real data   (recommended)
python run.py train --source real       # real only
python run.py train --source synthetic  # synthetic only (sanity check)
```
Prints cross-validated accuracy, per-class precision/recall, saves confusion matrices and the
tap-only / probe-only / combined ablation to `models/<version>/` (also visible on the MODEL tab, which also
has a *Retrain* button). The newest model is picked up at the next `demo` start (or via the MODEL tab).

### 8. Demo
```bash
python run.py demo --port COM5
```
Backup if hardware misbehaves: `python run.py demo --sim`.

## Where things are

| Path | What |
|---|---|
| `config.yaml` | materials, zones, sample rate, thresholds, probe, ML settings |
| `safety_rules.yaml` | safety rules (editable on the SAFETY tab) |
| `protocol/` | protocol spec + golden byte vectors |
| `firmware/spine_node/` | ESP32 firmware (`.ino` + header) |
| `spine/` | coordinator, functions, simulator, ML, apps, dashboard server |
| `data/synthetic/SYNTHETIC_dataset.npz` | 2400 synthetic events (300 per material×zone) |
| `data/real/` | your recordings · `data/simulated_recordings/` simulated ones (never mixed in) |
| `profiles/<user>.json` | per-user calibration |
| `models/<version>/` | trained models + reports + images |
| `docs/ARCHITECTURE.md`, `DECISIONS.md`, `WIRING.md` | design docs |

Commands: `python run.py demo|record|train|monitor|gen-data|ports|test` (+ `.bat` wrappers).

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No SPINE node found" / wrong COM port | `python run.py ports`; pass `--port COMx`; close Arduino Serial Monitor (it holds the port); try another cable (charge-only cables have no data); press EN on the board. In the dashboard: SPINE NODE tab → port → *Connect & start*. |
| Board resets when the app connects | Normal for ESP32 (DTR); the app waits ~2 s. |
| No signal / flat line | Check bias: both 1 MΩ present and 3V3/GND connected; ADC pins are GPIO34/35 (ESP32) or 1/2 (S3); piezo polarity doesn't matter. `monitor` shows idle counts. |
| Constant false triggers | Raise threshold (CALIBRATION step 1 does it, or SPINE NODE → SETUP_SENSOR); strap the band tighter; keep cables still (they act as microphones). |
| No triggers on taps | Lower threshold; check that bars move in `monitor`; check piezo contact with skin. |
| Strong 50 Hz hum | Shorten ADC leads, twist pairs, keep away from the laptop charger / unplug it. |
| Everything classified "unknown" | Model not trained on real data: record + `train --source mixed`. Lower `ml.unknown_threshold` in config.yaml. |
| Probe never fires / no probe data | SPINE NODE → SETUP_PROBE enabled; the chirp plays ~100 ms after each tap. |
| Dashboard blank | Hard refresh (Ctrl+F5); check terminal for Python errors. |
| `crc errors` rising on SPINE NODE tab | Serial noise / loose cable; frames are dropped, system keeps running. |

## Status — what is and is not verified

Verified here (see `python run.py test`): protocol encode/decode + golden vectors, parser resync/CRC,
simulator, features, training (synthetic/real/mixed + ablation), end-to-end SIMULATED pipeline, safety
rules, recorder, calibration + adaptation, serial transport (loopback), dashboard live in a browser.

**NOT verified (no hardware and no C/C++ compiler available while building):**
* Firmware has never been compiled or run on a board. `firmware/test/test_protocol.cpp` (host C++ vs golden
  vectors) was not run — run it once with `g++ -std=c++11 firmware/test/test_protocol.cpp -o t && ./t`.
* ADC timing at 10 kHz (analogRead speed) and real signal levels; the default threshold (120) is a guess.
* Real-world classification accuracy: unknown until real data is recorded.
