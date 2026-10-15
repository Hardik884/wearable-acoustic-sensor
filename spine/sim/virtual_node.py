"""A virtual node that speaks the same protocol as the ESP32 firmware (SIMULATED data)."""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .. import protocol as P
from ..config import CFG, window_sizes
from .generator import EventSynth, VirtualUser

FW_VERSION = 0x0100
LIVE_PERIOD = 0.05          # s between LIVE frames
LIVE_RATE = 1000            # Hz after decimation


class VirtualNode:
    def __init__(self, cfg: dict | None = None, node_id: int = 1, seed: int | None = None,
                 auto_taps: bool = True, realtime: bool = True):
        self.cfg = cfg or CFG
        n = self.cfg["node"]
        self.node_id = node_id
        self.rate = n["sample_rate"]
        self.threshold = n["threshold"]
        self.refractory_ms = n["refractory_ms"]
        self.probe = dict(n["probe"])
        self.running = False
        self.live = False
        self.auto_taps = auto_taps
        self.realtime = realtime
        self.synth = EventSynth(self.cfg, seed=seed if seed is not None else 123, rate=self.rate)
        self.user = VirtualUser.random(self.synth.rng)
        self.out: "queue.Queue[bytes]" = queue.Queue()
        self.parser = P.FrameParser()
        self.seq = 0
        self.hint: tuple[str, str] | None = None
        self._tap_requests: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._due: list[tuple[float, bytes]] = []
        self._live_buf = np.zeros((2, 0))
        self._next_tap = time.time() + 1.0
        self._last_event = 0.0
        self._t0 = time.time()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # -------------------------------------------------------------- lifecycle
    def start_thread(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="virtual-node")
            self._thread.start()

    def stop_thread(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------- protocol IO
    def _send(self, mtype: int, payload: bytes = b"") -> None:
        self.seq = (self.seq + 1) & 0xFFFF
        self.out.put(P.encode_frame(mtype, payload, self.node_id, self.seq))

    def _ack(self, ref: int, text: str = "ok", code: int = P.MsgCode.ACK) -> None:
        self._send(P.MsgType.SERVICE_MESSAGE, P.enc_message(code, ref, text))

    def _advert(self) -> bytes:
        p = self.probe
        return P.enc_advert(2, 1, self.rate, FW_VERSION, 12, 255, self.threshold, self.refractory_ms,
                            p["enabled"], p["f_start"], p["f_end"], p["duration_ms"])

    def feed(self, data: bytes) -> None:
        for f in self.parser.feed(data):
            self._handle(f)

    def _handle(self, f: P.Frame) -> None:
        M = P.MsgType
        with self._lock:
            if f.msg_type == M.SERVICE_DISCOVERY:
                self._send(M.SERVICE_ADVERTISEMENT, self._advert())
            elif f.msg_type == M.SETUP_SENSOR:
                d = P.dec_setup_sensor(f.payload)
                if d["rate_hz"] not in (5000, 10000, 20000):
                    return self._ack(f.msg_type, "bad rate", P.MsgCode.ERROR)
                self.rate, self.threshold, self.refractory_ms = d["rate_hz"], d["threshold"], d["refractory_ms"]
                self.synth = EventSynth(self.cfg, seed=int(self.synth.rng.integers(1 << 30)), rate=self.rate)
                self._ack(f.msg_type)
            elif f.msg_type == M.SETUP_PROBE:
                d = P.dec_setup_probe(f.payload)
                self.probe.update(enabled=d["enabled"], f_start=d["f_start"], f_end=d["f_end"],
                                  duration_ms=d["duration_ms"])
                self.synth.probe_cfg = self.probe
                self._ack(f.msg_type)
            elif f.msg_type == M.CALIBRATION_UPDATE:
                d = P.dec_calibration(f.payload)
                self.threshold, self.refractory_ms = d["threshold"], d["refractory_ms"]
                self._ack(f.msg_type)
            elif f.msg_type == M.START:
                self.live = P.dec_start(f.payload)["live"]
                self.running = True
                self._next_tap = time.time() + 1.0
                self._ack(f.msg_type, "started")
            elif f.msg_type == M.STOP:
                self.running = False
                self._ack(f.msg_type, "stopped")
            elif f.msg_type == M.RESET:
                self.running = False
                n = self.cfg["node"]
                self.rate, self.threshold, self.refractory_ms = n["sample_rate"], n["threshold"], n["refractory_ms"]
                self.probe = dict(n["probe"])
                self.synth = EventSynth(self.cfg, seed=int(self.synth.rng.integers(1 << 30)), rate=self.rate)
                self._ack(f.msg_type, "reset")

    # -------------------------------------------------------------- simulation
    def request_tap(self, material: str, zone: str) -> None:
        self._tap_requests.put((material, zone))

    def _make_tap(self, material: str, zone: str) -> None:
        now = time.time()
        if (now - self._last_event) * 1000 < self.refractory_ms:
            return
        # material "none" = light finger tap with nothing left touching (probe baseline)
        tap_m = "finger" if material == "none" else material
        ev = self.synth.synth(tap_m, zone, self.user, probe=bool(self.probe["enabled"]),
                              probe_material="none" if material == "none" else None)
        peak = float(np.abs(ev["ch"]).max())
        if peak < self.threshold:
            return                                    # too weak: node would not trigger
        self._last_event = now
        n_pre = ev["n_pre"]
        tap = ev["ch"][:, n_pre:]
        if self._live_buf.shape[1] < tap.shape[1]:
            self._live_buf = np.pad(self._live_buf, ((0, 0), (0, tap.shape[1] - self._live_buf.shape[1])))
        self._live_buf[:, : tap.shape[1]] += tap
        flags = P.FLAG_SIMULATED | (P.FLAG_PROBE_USED if ev["probe"] is not None else 0)
        payload = P.enc_data(int((now - self._t0) * 1000), self.rate, flags, ev["ch"], ev["probe"], n_pre,
                             self.threshold)
        # DATA leaves the node after the tap window + probe have been captured
        delay = 0.1 + (self.probe["duration_ms"] + self.probe["tail_ms"]) / 1000 + 0.02
        self._due.append((now + delay, payload))

    def _run(self) -> None:
        while not self._stop.is_set():
            t_start = time.time()
            self.tick(t_start)
            if self.realtime:
                time.sleep(max(0.0, LIVE_PERIOD - (time.time() - t_start)))
            else:
                time.sleep(0.001)

    def tick(self, now: float) -> None:
        with self._lock:
            if self.running:
                while not self._tap_requests.empty():
                    m, z = self._tap_requests.get()
                    self._make_tap(m, z)
                if self.auto_taps and now >= self._next_tap:
                    if self.hint and self.hint[0] and self.hint[1]:
                        m, z = self.hint
                        self._next_tap = now + 1.2
                    else:
                        rng = self.synth.rng
                        m = self.cfg["materials"][int(rng.integers(len(self.cfg["materials"])))]
                        z = self.cfg["zones"][int(rng.integers(len(self.cfg["zones"])))]
                        self._next_tap = now + float(rng.uniform(2.0, 4.0))
                    self._make_tap(m, z)
                if self.live:
                    self._emit_live(now)
            due = [d for d in self._due if d[0] <= now]
            self._due = [d for d in self._due if d[0] > now]
        if self.running:
            for _, payload in due:
                self._send(P.MsgType.DATA, payload)

    def _emit_live(self, now: float) -> None:
        n = int(self.rate * LIVE_PERIOD)
        if self._live_buf.shape[1] < n:
            self._live_buf = np.pad(self._live_buf, ((0, 0), (0, n - self._live_buf.shape[1])))
        chunk = self._live_buf[:, :n] + self.synth.idle(n, self.user)
        self._live_buf = self._live_buf[:, n:]
        dec = max(1, self.rate // LIVE_RATE)
        m = (n // dec) * dec
        blocks = chunk[:, :m].reshape(2, -1, dec)
        idx = np.abs(blocks).argmax(axis=2)
        live = np.take_along_axis(blocks, idx[..., None], axis=2)[..., 0]   # peak-preserving
        self._send(P.MsgType.LIVE, P.enc_live(int((now - self._t0) * 1000), self.rate // dec, live))
