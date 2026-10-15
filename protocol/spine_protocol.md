# SPINE-style serial protocol (v1)

Single source of truth. Implemented in `spine/protocol.py` (Python) and
`firmware/spine_node/spine_protocol.h` (C++). `protocol/golden_vectors.json` holds sample
messages encoded by Python; `tests/test_protocol.py` and `firmware/test/test_protocol.cpp`
both check against it.

All integers little-endian. Samples are int16 ADC counts with DC (mid-rail) removed.

## Frame
| field   | size | note |
|---------|------|------|
| sync    | 2    | `A5 5A` |
| version | 1    | protocol version (1) |
| type    | 1    | message type |
| node id | 1    | 0xFF = broadcast |
| seq     | 2    | sender sequence number |
| length  | 2    | payload bytes (max 16384) |
| payload | n    | |
| crc16   | 2    | CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over version..payload |

## Coordinator -> Node
| type | name | payload |
|------|------|---------|
| 0x01 | SERVICE_DISCOVERY    | empty |
| 0x02 | SETUP_SENSOR         | `u16 rate_hz, u16 threshold, u16 refractory_ms` (rate: 5000/10000/20000) |
| 0x03 | SETUP_PROBE          | `u8 enabled, u16 f_start_hz, u16 f_end_hz, u16 duration_ms` |
| 0x04 | CALIBRATION_UPDATE   | `u16 threshold, u16 refractory_ms` |
| 0x05 | START                | `u8 live` (1 = also stream LIVE frames) |
| 0x06 | STOP                 | empty |
| 0x07 | RESET                | empty (restores defaults, stops) |

## Node -> Coordinator
| type | name | payload |
|------|------|---------|
| 0x81 | SERVICE_ADVERTISEMENT | `u8 n_sensors, u8 n_actuators, u16 rate_hz, u16 fw_version(major<<8\|minor), u8 adc_bits, u8 board(0=ESP32,1=S3,255=sim), u16 threshold, u16 refractory_ms, u8 probe_enabled, u16 f_start, u16 f_end, u16 duration_ms` |
| 0x82 | DATA | header `u32 ts_ms, u16 rate_hz, u8 n_ch, u8 flags, u16 n_pre, u16 n_tap, u16 n_probe, u16 trig_level` then `n_ch*(n_pre+n_tap)` int16 (channel-major), then `n_ch*n_probe` int16 (channel-major). flags: bit0 probe used, bit1 SIMULATED |
| 0x83 | LIVE | `u32 ts_ms, u16 rate_hz, u8 n_ch, u8 n` then `n_ch*n` int16 (channel-major, peak-preserving decimation) |
| 0x84 | SERVICE_MESSAGE | `u8 code(0 ack,1 error,2 status), u8 ref_type, utf8 text` |

Channels: 0 = mic_0 (wrist), 1 = mic_1 (5 cm toward elbow).

## Session flow
1. Coordinator sends SERVICE_DISCOVERY; node answers SERVICE_ADVERTISEMENT.
2. Coordinator sends SETUP_SENSOR, SETUP_PROBE (each acked by SERVICE_MESSAGE).
3. START -> node streams DATA (one per tap) and, if live=1, LIVE (20 frames/s).
4. CALIBRATION_UPDATE may be sent any time. STOP / RESET.
