"""SPINE coordinator: discovers/configures the node, receives frames, runs the function
pipeline, and dispatches to apps through a listener API.

Apps implement `Listener` (on_new_node / on_data / on_live / on_service_message) and
never touch the transport -- they cannot tell SERIAL from SIMULATED.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from . import protocol as P
from .config import CFG
from .functions.library import Env, TapEvent, build_pipeline, run_pipeline
from .transports.base import Transport


class Listener:
    def on_new_node(self, info: dict) -> None: ...
    def on_data(self, event: TapEvent) -> None: ...
    def on_live(self, live: dict) -> None: ...
    def on_service_message(self, msg: dict) -> None: ...


class Coordinator:
    def __init__(self, transport: Transport, cfg: dict | None = None, node_id: int = 1):
        self.transport = transport
        self.cfg = cfg or CFG
        self.node_id = node_id
        self.listeners: list[Listener] = []
        self.env = Env()
        self.pipeline = build_pipeline(None, unknown_threshold=self.cfg["ml"]["unknown_threshold"])
        self.node_info: dict | None = None
        self.node_config: dict = {}
        self.running = False
        self.live_enabled = True
        self.parser = P.FrameParser()
        self.seq = 0
        self.event_times: deque = deque(maxlen=500)
        self.n_events = 0
        self.last_frame = 0.0
        self._advert = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------- listeners
    def add_listener(self, l: Listener) -> None:
        self.listeners.append(l)

    def _emit(self, name: str, *args) -> None:
        for l in self.listeners:
            try:
                getattr(l, name)(*args)
            except Exception as e:  # one broken app must not kill the stream
                print(f"[coordinator] listener {type(l).__name__}.{name} failed: {e!r}")

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.transport.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True, name="coordinator-rx")
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        try:
            self.send(P.MsgType.STOP)
        except Exception:
            pass
        time.sleep(0.05)
        self.transport.close()

    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                data = self.transport.read(0.1)
            except Exception as e:
                print("[coordinator] transport read error:", e)
                time.sleep(0.5)
                continue
            if not data:
                continue
            for f in self.parser.feed(data):
                self._on_frame(f)

    # ----------------------------------------------------------- tx helpers
    def send(self, msg_type: int, payload: bytes = b"") -> None:
        with self._lock:
            self.seq = (self.seq + 1) & 0xFFFF
            self.transport.write(P.encode_frame(msg_type, payload, self.node_id, self.seq))

    def discover(self, timeout: float = 4.0) -> dict | None:
        self._advert.clear()
        for _ in range(3):
            self.send(P.MsgType.SERVICE_DISCOVERY)
            if self._advert.wait(timeout / 3):
                return self.node_info
        return None

    def setup_sensor(self, rate_hz: int | None = None, threshold: int | None = None,
                     refractory_ms: int | None = None) -> None:
        n = self.cfg["node"]
        cur = self.node_info or {}
        rate_hz = rate_hz or cur.get("rate_hz") or n["sample_rate"]
        threshold = threshold or cur.get("threshold") or n["threshold"]
        refractory_ms = refractory_ms or cur.get("refractory_ms") or n["refractory_ms"]
        self.send(P.MsgType.SETUP_SENSOR, P.enc_setup_sensor(int(rate_hz), int(threshold), int(refractory_ms)))
        self.node_config.update(rate_hz=int(rate_hz), threshold=int(threshold), refractory_ms=int(refractory_ms))

    def setup_probe(self, enabled: bool | None = None, f_start: int | None = None, f_end: int | None = None,
                    duration_ms: int | None = None) -> None:
        p = self.cfg["node"]["probe"]
        enabled = p["enabled"] if enabled is None else enabled
        f_start, f_end = f_start or p["f_start"], f_end or p["f_end"]
        duration_ms = duration_ms or p["duration_ms"]
        self.send(P.MsgType.SETUP_PROBE, P.enc_setup_probe(enabled, int(f_start), int(f_end), int(duration_ms)))
        self.node_config.update(probe_enabled=bool(enabled), f_start=int(f_start), f_end=int(f_end),
                                duration_ms=int(duration_ms))

    def calibration_update(self, threshold: int, refractory_ms: int) -> None:
        self.send(P.MsgType.CALIBRATION_UPDATE, P.enc_calibration(int(threshold), int(refractory_ms)))
        self.node_config.update(threshold=int(threshold), refractory_ms=int(refractory_ms))

    def start_node(self, live: bool = True) -> None:
        self.live_enabled = live
        self.send(P.MsgType.START, P.enc_start(live))
        self.running = True

    def stop_node(self) -> None:
        self.send(P.MsgType.STOP)
        self.running = False

    def reset_node(self) -> None:
        self.send(P.MsgType.RESET)
        self.running = False

    def connect_and_start(self, live: bool = True) -> dict | None:
        """Default bring-up: discover -> configure from config.yaml -> START."""
        info = self.discover()
        if info is None:
            return None
        n = self.cfg["node"]
        self.setup_sensor(n["sample_rate"], n["threshold"], n["refractory_ms"])
        p = n["probe"]
        self.setup_probe(p["enabled"], p["f_start"], p["f_end"], p["duration_ms"])
        time.sleep(0.15)
        self.start_node(live)
        return info

    def sim_hint(self, material: str | None, zone: str | None) -> None:
        self.transport.set_hint(material, zone)

    # ----------------------------------------------------------- processing
    def set_models(self, bundle) -> None:
        self.env.models = bundle

    def set_profile(self, profile: dict | None) -> None:
        self.env.profile = profile

    def process(self, raw: dict, node_id: int = 0) -> TapEvent:
        ev = TapEvent(raw=raw, ts=time.time(), node_id=node_id,
                      env=Env(profile=self.env.profile, models=self.env.models))
        return run_pipeline(self.pipeline, ev)

    def events_per_min(self) -> int:
        now = time.time()
        return sum(1 for t in self.event_times if now - t <= 60)

    def status(self) -> dict:
        return dict(transport=self.transport.name, simulated=self.transport.is_simulated,
                    description=self.transport.description, connected=self.node_info is not None,
                    running=self.running, events=self.n_events, events_per_min=self.events_per_min(),
                    crc_errors=self.parser.crc_errors, last_frame_age=(time.time() - self.last_frame)
                    if self.last_frame else None, node=self.node_info, config=self.node_config)

    def _on_frame(self, f: P.Frame) -> None:
        self.last_frame = time.time()
        M = P.MsgType
        if f.msg_type == M.SERVICE_ADVERTISEMENT:
            self.node_info = P.decode(f)
            self.node_info["node_id"] = f.node_id
            self.node_config.update({k: self.node_info[k] for k in
                                     ("rate_hz", "threshold", "refractory_ms", "probe_enabled", "f_start",
                                      "f_end", "duration_ms")})
            self._advert.set()
            self._emit("on_new_node", self.node_info)
        elif f.msg_type == M.DATA:
            raw = P.decode(f)
            ev = self.process(raw, f.node_id)
            self.n_events += 1
            self.event_times.append(time.time())
            self._emit("on_data", ev)
        elif f.msg_type == M.LIVE:
            self._emit("on_live", P.decode(f))
        elif f.msg_type == M.SERVICE_MESSAGE:
            self._emit("on_service_message", P.decode(f))
