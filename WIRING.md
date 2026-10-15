# Wiring

Three piezo discs, six 1 MΩ resistors, one ESP32 / ESP32-S3, USB cable. Nothing else.

## Pin map (ADC1 only)

| Function | ESP32 (classic) | ESP32-S3 | Notes |
|---|---|---|---|
| mic_0 (wrist)             | **GPIO34** | **GPIO1** | ADC1, input only on ESP32 |
| mic_1 (5 cm toward elbow) | **GPIO35** | **GPIO2** | ADC1 |
| act_0 drive A             | **GPIO25** | **GPIO15** | pin must be < 32 (fast register write) |
| act_0 drive B (inverted)  | **GPIO26** | **GPIO16** | |
| 3V3 / GND                 | 3V3 / GND  | 3V3 / GND | |

The pins are `#define`d at the top of `firmware/spine_node/spine_node.ino`. Board is auto-detected
(`BOARD_IS_S3`); to force the S3 map add `#define BOARD_IS_S3 1` above the pin block.

## Circuit (per microphone — build two)

```
            3V3
             |
           [1 MΩ]
             |
             +----------- ADC pin (GPIO34 / GPIO35)   <- mid-rail ≈ 1.65 V
             |
           [1 MΩ]            piezo disc
             |            (+ brass side)   (- ceramic side)
            GND                 |               |
                                +--- to ADC pin |
                                              GND
```

* The two 1 MΩ resistors make a 1.65 V bias at the ADC pin; the piezo sits between the ADC pin and GND,
  so a tap swings the pin above/below mid-rail (AC-coupled by the piezo's own capacitance).
* Keep the ADC-side wire short; twist/shield it if you see mains hum (the dashboard shows 50 Hz as a
  steady buzz on the live waveform).

## Actuator (no extra parts)

```
 GPIO25 ────────────┐
                    │  piezo disc (act_0)
 GPIO26 ────────────┘
```

The two GPIOs are driven in opposite phase, so the disc sees ±3.3 V (6.6 V peak-to-peak). Both pins
rest LOW when idle. Mount the actuator between/next to the mics so it couples into the same skin patch.

## Placement

```
  wrist ─────────────────────────────────────────────► elbow
     [mic_0]   (act_0)        [mic_1]
        |<------- ~5 cm ------->|
   zone "wrist"             zone "mid_forearm"
```

Press the discs against the skin with a strap/band, ceramic side outward (brass side on skin gives a
better signal; either works — be consistent between calibration, recording and demo).

## Assumptions to verify on the bench

1. Idle ADC reading is ≈ 1900–2200 on both mics (run `python run.py monitor`).
2. A finger tap on each mic shows a spike on that channel first.
3. Actuator: with probe enabled you should hear/feel a short chirp ~100 ms after each tap.
4. ADC input 0–3.1 V range is configured with 11 dB attenuation in firmware.
