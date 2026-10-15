# Design decisions (made autonomously, tonight)

| # | Decision | Why |
|---|---|---|
| 1 | **10 kHz per channel**, chirp 300 Hz → 4 kHz (< 5 kHz Nyquist) | `analogRead` of 2 channels ≈ 40–60 µs on both boards; 10 kHz leaves 100 µs per sample. 20 kHz is selectable but may drop samples (node counts overruns). Needs retraining if changed. |
| 2 | One 20 kHz hardware timer paces both the ADC (every 2nd tick) and the actuator square-wave chirp (integer phase accumulator, no FPU in ISR) | A single timer is simplest and keeps ADC/actuator phase-locked. |
| 3 | Actuator is a **square** chirp, two GPIOs in opposite phase | No DAC / extra parts; harmonics are fine because the arm acts as the filter. |
| 4 | Node subtracts a slowly-tracked DC (mid-rail) and sends **int16 deviations** | Smaller numbers, no bias dependence on the host. Trigger is on deviation > threshold (default 120 counts). |
| 5 | Frame = `A5 5A | ver | type | node | seq u16 | len u16 | payload | CRC-16/CCITT` | Resync-able, 11 bytes overhead, same layout both languages. Golden byte vectors checked in Python; C++ test is provided (`firmware/test`). |
| 6 | Event window: **20 ms pre + 100 ms tap + 57 ms probe response** (3 ms of direct coupling skipped, 10 ms ring-down kept) | Matches the spec; 1770 samples/channel ≈ 7 KB per DATA frame, ≈ 80 ms at 921600 baud. |
| 7 | Channel 0 = wrist mic, channel 1 = mic toward elbow. Zones: `wrist` (tap near mic_0), `mid_forearm` (tap near/after mic_1). TDOA sign: positive = mic_1 later. | Only two zones are required; the geometry gives a clear TDOA sign flip. |
| 8 | Classifier default **RandomForest** (config `ml.model: svm` switches) | Robust with few real samples, no scaling issues, native probabilities for the "unknown" threshold. |
| 9 | CV for synthetic data is **GroupKFold by virtual user**; real data uses StratifiedKFold | Honest estimate of generalisation to unseen users. Still a synthetic number — labelled as such everywhere. |
| 10 | "unknown" when top-class probability < 0.5 (`ml.unknown_threshold`) | Simple, tunable. |
| 11 | Material model uses `tap_*` + `probe_*` features; zone model uses cross-channel (`xc_*`) + per-channel level/centroid features | Material is object-dependent, zone is geometry-dependent; avoids leaking one into the other. |
| 12 | Probe baseline normalisation is applied at **feature level inside F_PREPROCESS**: `band_dB − user_baseline + model_reference_baseline` | Keeps models trained on different users comparable. Reference = nominal synthetic baseline (synthetic/mixed) or mean of recorded baselines (real). |
| 13 | Simulated recordings go to `data/simulated_recordings/`, never `data/real/` | Prevents synthetic taps being mistaken for real training data. |
| 14 | Personal adaptation: retrain on synthetic + user samples (weight 5); reported before/after accuracy is on a held-out half of the user's samples | Quick (~10 s) and honest; final model uses all samples. |
| 15 | Dashboard is one dependency-free HTML page (no CDN) with canvas charts | Must work offline at the venue / on a projector. |
| 16 | No BLE tonight; `Transport` interface is the only thing apps/coordinator see | A `BleTransport` can be added without touching apps. |
| 17 | `SIMULATED` virtual node is a full protocol peer (frames over the same parser), taps timed in real time | Backup demo exercises the exact code path of the real device. |
| 18 | C++ was **not compiled** in this environment (no compiler installed) | See README "Not verified". Everything in firmware is written for Arduino-ESP32 core 2.x and 3.x. |
