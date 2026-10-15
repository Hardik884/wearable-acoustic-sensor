import json

import numpy as np

from spine import protocol as P
from spine.golden import GOLDEN_PATH, build_samples


def test_golden_vectors_match():
    """Python encoder output must equal the committed vectors the C++ test also checks."""
    golden = json.loads(GOLDEN_PATH.read_text())
    for name, data in build_samples().items():
        assert data.hex() == golden[name], name


def test_roundtrip_data():
    ch = np.random.default_rng(0).integers(-2000, 2000, (2, 300))
    pr = np.random.default_rng(1).integers(-2000, 2000, (2, 50))
    frames = P.FrameParser().feed(P.encode_frame(P.MsgType.DATA, P.enc_data(77, 10000, 1, ch, pr, 20), 3, 9))
    assert len(frames) == 1 and frames[0].node_id == 3 and frames[0].seq == 9
    d = P.decode(frames[0])
    assert d["n_pre"] == 20 and d["n_tap"] == 280 and d["probe_used"]
    assert np.array_equal(d["ch"], ch) and np.array_equal(d["probe"], pr)


def test_parser_resync_and_crc():
    good = P.encode_frame(P.MsgType.STOP, b"", 1, 1)
    bad = bytearray(P.encode_frame(P.MsgType.SETUP_SENSOR, P.enc_setup_sensor(1, 2, 3), 1, 2))
    bad[-3] ^= 0xFF
    parser = P.FrameParser()
    out = parser.feed(b"\x00\x11garbage" + bytes(bad) + good + good[:5])
    out += parser.feed(good[5:])
    assert [f.msg_type for f in out] == [P.MsgType.STOP, P.MsgType.STOP]
    assert parser.crc_errors == 1


def test_advert_roundtrip():
    p = P.enc_advert(2, 1, 10000, 0x0102, 12, 1, 150, 300, True, 300, 4000, 50)
    d = P.dec_advert(p)
    assert d["firmware"] == "1.2" and d["board_name"] == "ESP32-S3" and d["probe_enabled"]
