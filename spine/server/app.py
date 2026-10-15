"""FastAPI + WebSocket dashboard backend. A DashboardApp is just another SPINE listener."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..apps import calibration as C
from ..apps.recorder import RecorderApp
from ..apps.safety import SafetyApp
from ..apps.store import Store
from ..config import CFG, MODELS
from ..coordinator import Coordinator, Listener
from ..ml.model import ModelBundle
from ..transports.serial_transport import SerialTransport, list_serial_ports
from ..transports.sim_transport import SimTransport

STATIC = Path(__file__).parent / "static"


def make_transport():
    mode = os.environ.get("SPINE_MODE", "sim").lower()
    if mode == "serial":
        return SerialTransport(os.environ.get("SPINE_PORT", "auto"))
    return SimTransport(seed=int(os.environ.get("SPINE_SEED", "5")))


class Hub:
    """Fan-out of JSON messages to all connected browsers (thread-safe publish)."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish(self, msg: dict) -> None:
        if not self.loop or not self.clients:
            return
        data = json.dumps(msg, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        asyncio.run_coroutine_threadsafe(self._send(data), self.loop)

    async def _send(self, data: str) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


class DashboardApp(Listener):
    """Turns SPINE events into UI messages and stores them."""

    def __init__(self, hub: Hub, store: Store, coord: Coordinator, state: "AppState"):
        self.hub, self.store, self.coord, self.state = hub, store, coord, state
        self._last_status = 0.0

    def on_new_node(self, info):
        self.hub.publish(dict(type="node", node=info, status=self.coord.status()))

    def on_service_message(self, msg):
        self.hub.publish(dict(type="message", **msg))

    def on_live(self, live):
        self.hub.publish(dict(type="live", rate=live["rate_hz"], samples=live["samples"].astype(int).tolist()))
        if time.time() - self._last_status > 1.0:
            self._last_status = time.time()
            self.hub.publish(dict(type="status", status=self.coord.status()))

    def on_data(self, ev):
        d = ev.to_dict()
        d["user"] = (self.coord.env.profile or {}).get("name", "default")
        d["id"] = self.store.add_event(d, d["user"])
        fs, on = ev.rate, ev.onset
        a = max(on - int(0.002 * fs), 0)
        seg = ev.ch[:, a: a + int(0.06 * fs)]
        d["wave"] = np.rint(seg[:, ::2]).astype(int).tolist()
        d["wave_dt_ms"] = 2000.0 / fs
        s = ev.ch[:, max(on - 2, 0): max(on - 2, 0) + int(0.05 * fs)]
        mag = np.abs(np.fft.rfft(s * np.hanning(s.shape[1]), 1024, axis=1))
        db = 20 * np.log10(mag + 1.0)
        d["spec"] = db[:, :512].reshape(2, 64, 8).mean(axis=2).round(1).tolist()
        d["spec_fmax"] = fs / 2
        d["type"] = "event"
        self.hub.publish(d)
        self.hub.publish(dict(type="status", status=self.coord.status()))


class AppState:
    def __init__(self):
        self.hub = Hub()
        self.store = Store()
        self.transport = make_transport()
        self.coord = Coordinator(self.transport)
        self.model_job: dict = dict(running=False, message="", source=None)
        self.dashboard = DashboardApp(self.hub, self.store, self.coord, self)
        self.safety = SafetyApp(self.store, emit=lambda a: self.hub.publish(dict(type="alert", alert=a)))
        self.recorder = RecorderApp(self.coord, emit=self.hub.publish)
        self.calib = C.CalibrationApp(self.coord, emit=self.hub.publish)
        for l in (self.dashboard, self.safety, self.recorder, self.calib):
            self.coord.add_listener(l)

    def bring_up(self):
        mb = ModelBundle.latest()
        if mb is None:
            from ..ml.train import ensure_model
            mb = ensure_model()
        self.coord.set_models(mb)
        self.hub.publish(dict(type="model", model=self.model_state()))
        info = None
        try:
            self.coord.start()
            info = self.coord.connect_and_start(live=True)
        except Exception as e:
            print("[server] could not start transport:", e)
        if info is None:
            print("[server] node not found yet. Use the SPINE NODE tab > Discover.")
        else:
            print(f"[server] node: {info['board_name']} fw {info['firmware']} @ {info['rate_hz']} Hz")

    def model_state(self) -> dict:
        mb = self.coord.env.models
        meta = mb.meta if mb else None
        tags = sorted([p.name for p in MODELS.glob("*") if (p / "meta.json").exists()], reverse=True)
        return dict(active=meta, tags=tags, job=self.model_job)


def create_app() -> FastAPI:
    st = AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        st.hub.loop = asyncio.get_running_loop()
        threading.Thread(target=st.bring_up, daemon=True).start()
        yield
        st.coord.close()

    app = FastAPI(title="Wearable Acoustic Sensor", lifespan=lifespan)
    app.state.st = st
    MODELS.mkdir(exist_ok=True)
    app.mount("/models", StaticFiles(directory=str(MODELS)), name="models")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    # ------------------------------------------------------------ state / ws
    def full_state() -> dict:
        return dict(status=st.coord.status(), config=CFG, model=st.model_state(),
                    profile=st.coord.env.profile, recorder=st.recorder.state(), calib=st.calib.state(),
                    rules=st.safety.rules, ports=list_serial_ports() if not st.transport.is_simulated else [],
                    alerts=st.store.alerts(50))

    @app.get("/api/state")
    def api_state():
        return full_state()

    @app.websocket("/ws")
    async def ws(ws: WebSocket):
        await ws.accept()
        st.hub.clients.add(ws)
        try:
            await ws.send_text(json.dumps(dict(type="hello", state=full_state()), default=str))
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            st.hub.clients.discard(ws)

    # ------------------------------------------------------------ node / SPINE
    @app.post("/api/node/{action}")
    def node_action(action: str, body: dict = Body(default={})):
        c = st.coord
        if action == "connect":                  # (re)open the serial port, discover, configure, start
            if not st.transport.is_simulated and body.get("port"):
                st.transport.port = body["port"]
            try:
                c.start()
            except Exception as e:
                raise HTTPException(502, f"cannot open transport: {e}")
            if c.connect_and_start(live=True) is None:
                raise HTTPException(504, "Port opened but no SERVICE_ADVERTISEMENT (wrong port / firmware?)")
        elif action == "discover":
            info = c.discover()
            if info is None:
                raise HTTPException(504, "No SERVICE_ADVERTISEMENT received")
        elif action == "setup":
            if "rate_hz" in body or "threshold" in body or "refractory_ms" in body:
                c.setup_sensor(body.get("rate_hz"), body.get("threshold"), body.get("refractory_ms"))
            if "probe_enabled" in body:
                c.setup_probe(body.get("probe_enabled"), body.get("f_start"), body.get("f_end"),
                              body.get("duration_ms"))
        elif action == "start":
            c.start_node(live=bool(body.get("live", True)))
        elif action == "stop":
            c.stop_node()
        elif action == "reset":
            c.reset_node()
        else:
            raise HTTPException(404, action)
        time.sleep(0.2)
        return c.status()

    @app.post("/api/sim/tap")
    def sim_tap(body: dict = Body(...)):
        return dict(ok=st.transport.simulate_tap(body["material"], body["zone"]))

    # ------------------------------------------------------------ events / analytics
    @app.get("/api/events")
    def events(limit: int = 200, material: str | None = None, zone: str | None = None):
        return st.store.events(limit, material or None, zone or None)

    @app.get("/api/events.csv")
    def events_csv():
        return PlainTextResponse(st.store.to_csv(st.store.events(100000)), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=events.csv"})

    @app.post("/api/events/clear")
    def events_clear():
        st.store.clear_events()
        return dict(ok=True)

    @app.get("/api/analytics")
    def analytics():
        return st.store.analytics()

    # ------------------------------------------------------------ safety
    @app.get("/api/safety/rules")
    def rules():
        return st.safety.rules

    @app.put("/api/safety/rules")
    def put_rules(rules: list[dict] = Body(...)):
        try:
            st.safety.set_rules(rules)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        return st.safety.rules

    @app.get("/api/alerts")
    def alerts():
        return st.store.alerts(300)

    @app.post("/api/alerts/ack")
    def ack_all(body: dict = Body(default={})):
        st.store.ack_alert(body.get("id"))
        st.hub.publish(dict(type="alerts_acked", id=body.get("id")))
        return dict(ok=True)

    @app.get("/api/alerts.csv")
    def alerts_csv():
        return PlainTextResponse(st.store.to_csv(st.store.alerts(100000)), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=alerts.csv"})

    # ------------------------------------------------------------ calibration / profiles
    @app.get("/api/calib")
    def calib_state():
        return st.calib.state()

    @app.post("/api/calib/start")
    def calib_start(body: dict = Body(...)):
        st.calib.start(body.get("user", "default"))
        return st.calib.state()

    @app.post("/api/calib/step")
    def calib_step(body: dict = Body(...)):
        try:
            st.calib.run_step(body["step"])
        except ValueError:
            raise HTTPException(400, "unknown step")
        return st.calib.state()

    @app.post("/api/calib/adapt")
    def calib_adapt():
        st.calib.adapt()
        return st.calib.state()

    @app.post("/api/profiles/select")
    def select_profile(body: dict = Body(...)):
        name = C.safe_name(body["name"])
        prof = C.load_profile(name)
        if prof is None:
            raise HTTPException(404, "no such profile")
        st.calib.user = name
        st.calib.profile = prof
        st.calib.activate(prof)
        st.hub.publish(dict(type="profile", profile=prof))
        return prof

    @app.get("/api/profiles")
    def profiles():
        return dict(profiles=C.list_profiles(), active=(st.coord.env.profile or {}).get("name"))

    # ------------------------------------------------------------ recorder
    @app.get("/api/recorder")
    def rec_state():
        return st.recorder.state()

    @app.post("/api/recorder/start")
    def rec_start(body: dict = Body(...)):
        try:
            st.recorder.start(body["material"], body["zone"], body.get("target", 40),
                              (st.coord.env.profile or {}).get("name", "default"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return st.recorder.state()

    @app.post("/api/recorder/stop")
    def rec_stop():
        st.recorder.stop()
        return st.recorder.state()

    @app.post("/api/recorder/delete_last")
    def rec_del():
        return dict(deleted=st.recorder.delete_last(), state=st.recorder.state())

    # ------------------------------------------------------------ model
    @app.get("/api/model")
    def model():
        return st.model_state()

    @app.post("/api/model/retrain")
    def retrain(body: dict = Body(default={})):
        if st.model_job["running"]:
            raise HTTPException(409, "training already running")
        source = body.get("source", "synthetic")
        mtype = body.get("model_type") or CFG["ml"]["model"]
        st.model_job.update(running=True, message=f"training on {source} data...", source=source)
        st.hub.publish(dict(type="model", model=st.model_state()))

        def job():
            from ..ml.train import train
            try:
                tag = train(source, mtype, verbose=False)
                st.coord.set_models(ModelBundle.load_tag(tag))
                st.model_job.update(running=False, message=f"done: {tag}")
            except SystemExit as e:
                st.model_job.update(running=False, message=str(e))
            except Exception as e:
                st.model_job.update(running=False, message=f"failed: {e!r}")
            st.hub.publish(dict(type="model", model=st.model_state()))
        threading.Thread(target=job, daemon=True).start()
        return st.model_state()

    @app.post("/api/model/activate")
    def activate_model(body: dict = Body(...)):
        mb = ModelBundle.load_tag(body["tag"])
        if mb is None:
            raise HTTPException(404, "no such model")
        st.coord.set_models(mb)
        return st.model_state()

    return app
