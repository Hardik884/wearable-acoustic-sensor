"""Sample messages used to prove the C++ and Python encoders agree byte-for-byte.

`python -m spine.golden` rewrites protocol/golden_vectors.json.
firmware/spine_node/test_protocol.cpp builds the same messages and compares.
"""
import json
from pathlib import Path

import numpy as np

from . import protocol as P


def build_samples() -> dict[str, bytes]:
    ch = np.array([[(c + 1) * (i * 37 - 100) for i in range(10)] for c in range(2)])
    probe = np.array([[-(c + 1) * (i * 11 + 5) for i in range(3)] for c in range(2)])
    live = np.array([[(c * 7 + 1) * (i - 2) * 13 for i in range(5)] for c in range(2)])
    M = P.MsgType
    return {
        "discovery": P.encode_frame(M.SERVICE_DISCOVERY, b"", 0, 1),
        "setup_sensor": P.encode_frame(M.SETUP_SENSOR, P.enc_setup_sensor(10000, 150, 300), 1, 2),
        "setup_probe": P.encode_frame(M.SETUP_PROBE, P.enc_setup_probe(True, 300, 4000, 50), 1, 3),
        "calibration": P.encode_frame(M.CALIBRATION_UPDATE, P.enc_calibration(200, 350), 1, 4),
        "start": P.encode_frame(M.START, P.enc_start(True), 1, 5),
        "stop": P.encode_frame(M.STOP, b"", 1, 6),
        "reset": P.encode_frame(M.RESET, b"", 1, 7),
        "advert": P.encode_frame(
            M.SERVICE_ADVERTISEMENT,
            P.enc_advert(2, 1, 10000, 0x0100, 12, 0, 150, 300, True, 300, 4000, 50), 1, 8),
        "data": P.encode_frame(
            M.DATA, P.enc_data(123456, 10000, P.FLAG_PROBE_USED, ch, probe, n_pre=4, trig_level=150),
            1, 9),
        "live": P.encode_frame(M.LIVE, P.enc_live(5000, 1000, live), 1, 10),
        "message": P.encode_frame(M.SERVICE_MESSAGE, P.enc_message(P.MsgCode.ACK, M.START, "ok"), 1, 11),
    }


GOLDEN_PATH = Path(__file__).resolve().parent.parent / "protocol" / "golden_vectors.json"

if __name__ == "__main__":
    out = {k: v.hex() for k, v in build_samples().items()}
    GOLDEN_PATH.write_text(json.dumps(out, indent=1))
    print("wrote", GOLDEN_PATH)
