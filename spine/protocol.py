"""SPINE-style binary protocol (Python side). Spec: protocol/spine_protocol.md.

Frame (little-endian):
    0xA5 0x5A | ver u8 | type u8 | node u8 | seq u16 | len u16 | payload | crc16 u16
CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over ver..payload.
The C++ twin is firmware/spine_node/spine_protocol.h; both must give identical bytes
(see tests/test_protocol.py and protocol/golden_vectors.json).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

SYNC = b"\xA5\x5A"
VERSION = 1
HEADER_SIZE = 9
MAX_PAYLOAD = 16384
BROADCAST = 0xFF


class MsgType(IntEnum):
    # coordinator -> node
    SERVICE_DISCOVERY = 0x01
    SETUP_SENSOR = 0x02
    SETUP_PROBE = 0x03
    CALIBRATION_UPDATE = 0x04
    START = 0x05
    STOP = 0x06
    RESET = 0x07
    # node -> coordinator
    SERVICE_ADVERTISEMENT = 0x81
    DATA = 0x82
    LIVE = 0x83
    SERVICE_MESSAGE = 0x84


class MsgCode(IntEnum):  # SERVICE_MESSAGE.code
    ACK = 0
    ERROR = 1
    STATUS = 2


FLAG_PROBE_USED = 0x01
FLAG_SIMULATED = 0x02


def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass
class Frame:
    msg_type: int
    node_id: int
    seq: int
    payload: bytes
    version: int = VERSION


def encode_frame(msg_type: int, payload: bytes = b"", node_id: int = 0, seq: int = 0,
                 version: int = VERSION) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("payload too large")
    body = struct.pack("<BBBHH", version, int(msg_type), node_id, seq & 0xFFFF, len(payload)) + payload
    return SYNC + body + struct.pack("<H", crc16(body))


class FrameParser:
    """Incremental stream parser: resyncs on garbage, counts CRC errors."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.crc_errors = 0
        self.garbage_bytes = 0

    def feed(self, data: bytes) -> list[Frame]:
        self.buf += data
        out: list[Frame] = []
        while True:
            i = self.buf.find(SYNC)
            if i < 0:
                # keep a trailing 0xA5 which may be half a sync word
                keep = 1 if self.buf.endswith(SYNC[:1]) else 0
                self.garbage_bytes += len(self.buf) - keep
                del self.buf[: len(self.buf) - keep]
                return out
            if i > 0:
                self.garbage_bytes += i
                del self.buf[:i]
            if len(self.buf) < HEADER_SIZE:
                return out
            ver, mtype, node, seq, plen = struct.unpack_from("<BBBHH", self.buf, 2)
            if plen > MAX_PAYLOAD:
                del self.buf[:1]
                continue
            total = HEADER_SIZE + plen + 2
            if len(self.buf) < total:
                return out
            body = bytes(self.buf[2 : HEADER_SIZE + plen])
            (rx_crc,) = struct.unpack_from("<H", self.buf, HEADER_SIZE + plen)
            if crc16(body) != rx_crc:
                self.crc_errors += 1
                del self.buf[:1]
                continue
            out.append(Frame(mtype, node, seq, body[7:], ver))
            del self.buf[:total]


# ---------------------------------------------------------------- payload codecs
def enc_setup_sensor(rate_hz: int, threshold: int, refractory_ms: int) -> bytes:
    return struct.pack("<HHH", rate_hz, threshold, refractory_ms)


def dec_setup_sensor(p: bytes) -> dict:
    r, t, f = struct.unpack("<HHH", p)
    return dict(rate_hz=r, threshold=t, refractory_ms=f)


def enc_setup_probe(enabled: bool, f_start: int, f_end: int, duration_ms: int) -> bytes:
    return struct.pack("<BHHH", int(enabled), f_start, f_end, duration_ms)


def dec_setup_probe(p: bytes) -> dict:
    e, a, b, d = struct.unpack("<BHHH", p)
    return dict(enabled=bool(e), f_start=a, f_end=b, duration_ms=d)


def enc_calibration(threshold: int, refractory_ms: int) -> bytes:
    return struct.pack("<HH", threshold, refractory_ms)


def dec_calibration(p: bytes) -> dict:
    t, f = struct.unpack("<HH", p)
    return dict(threshold=t, refractory_ms=f)


def enc_start(live: bool) -> bytes:
    return struct.pack("<B", int(live))


def dec_start(p: bytes) -> dict:
    return dict(live=bool(p[0]) if p else False)


_ADV = "<BBHHBBHHBHHH"


def enc_advert(n_sensors, n_actuators, rate_hz, fw_version, adc_bits, board, threshold,
               refractory_ms, probe_enabled, f_start, f_end, duration_ms) -> bytes:
    return struct.pack(_ADV, n_sensors, n_actuators, rate_hz, fw_version, adc_bits, board,
                       threshold, refractory_ms, int(probe_enabled), f_start, f_end, duration_ms)


def dec_advert(p: bytes) -> dict:
    v = struct.unpack(_ADV, p)
    keys = ["n_sensors", "n_actuators", "rate_hz", "fw_version", "adc_bits", "board", "threshold",
            "refractory_ms", "probe_enabled", "f_start", "f_end", "duration_ms"]
    d = dict(zip(keys, v))
    d["probe_enabled"] = bool(d["probe_enabled"])
    d["firmware"] = f"{d['fw_version'] >> 8}.{d['fw_version'] & 0xFF}"
    d["board_name"] = {0: "ESP32", 1: "ESP32-S3", 255: "SIMULATED"}.get(d["board"], "?")
    return d


_DATA = "<IHBBHHHH"
_DATA_SIZE = struct.calcsize(_DATA)


def enc_data(ts_ms: int, rate_hz: int, flags: int, pre_tap: np.ndarray, probe: np.ndarray | None,
             n_pre: int, trig_level: int = 0) -> bytes:
    """pre_tap: (n_ch, n_pre+n_tap) int16-able; probe: (n_ch, n_probe) or None."""
    n_ch, n_pt = pre_tap.shape
    n_probe = 0 if probe is None else probe.shape[1]
    head = struct.pack(_DATA, ts_ms & 0xFFFFFFFF, rate_hz, n_ch, flags, n_pre, n_pt - n_pre,
                       n_probe, trig_level)
    body = np.clip(np.rint(pre_tap), -32768, 32767).astype("<i2").tobytes()
    if probe is not None:
        body += np.clip(np.rint(probe), -32768, 32767).astype("<i2").tobytes()
    return head + body


def dec_data(p: bytes) -> dict:
    ts, rate, n_ch, flags, n_pre, n_tap, n_probe, trig = struct.unpack_from(_DATA, p, 0)
    n_pt = n_pre + n_tap
    a = np.frombuffer(p, dtype="<i2", offset=_DATA_SIZE, count=n_ch * n_pt + n_ch * n_probe)
    ch = a[: n_ch * n_pt].reshape(n_ch, n_pt).astype(np.float32)
    probe = a[n_ch * n_pt :].reshape(n_ch, n_probe).astype(np.float32) if n_probe else None
    return dict(ts_ms=ts, rate_hz=rate, flags=flags, n_pre=n_pre, n_tap=n_tap, trig_level=trig,
                ch=ch, probe=probe, probe_used=bool(flags & FLAG_PROBE_USED),
                simulated=bool(flags & FLAG_SIMULATED))


_LIVE = "<IHBB"


def enc_live(ts_ms: int, rate_hz: int, samples: np.ndarray) -> bytes:
    n_ch, n = samples.shape
    return struct.pack(_LIVE, ts_ms & 0xFFFFFFFF, rate_hz, n_ch, n) + \
        np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()


def dec_live(p: bytes) -> dict:
    ts, rate, n_ch, n = struct.unpack_from(_LIVE, p, 0)
    a = np.frombuffer(p, dtype="<i2", offset=struct.calcsize(_LIVE), count=n_ch * n)
    return dict(ts_ms=ts, rate_hz=rate, samples=a.reshape(n_ch, n).astype(np.float32))


def enc_message(code: int, ref_type: int, text: str = "") -> bytes:
    return struct.pack("<BB", code, ref_type) + text.encode("utf-8")[:200]


def dec_message(p: bytes) -> dict:
    code, ref = struct.unpack_from("<BB", p, 0)
    return dict(code=code, code_name=MsgCode(code).name if code in MsgCode._value2member_map_ else str(code),
                ref_type=ref, text=p[2:].decode("utf-8", "replace"))


DECODERS = {
    MsgType.SETUP_SENSOR: dec_setup_sensor, MsgType.SETUP_PROBE: dec_setup_probe,
    MsgType.CALIBRATION_UPDATE: dec_calibration, MsgType.START: dec_start,
    MsgType.SERVICE_ADVERTISEMENT: dec_advert, MsgType.DATA: dec_data, MsgType.LIVE: dec_live,
    MsgType.SERVICE_MESSAGE: dec_message,
}


def decode(frame: Frame) -> dict:
    fn = DECODERS.get(frame.msg_type)
    return fn(frame.payload) if fn else {}
