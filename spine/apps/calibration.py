"""Calibration app: per-user wizard (silence -> gain -> probe baseline -> personal samples -> adapt).

Profiles live in profiles/<name>.json; personal samples in profiles/<name>_samples/.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
import time
from typing import Callable

import numpy as np

from .. import dataset as D
from ..config import CFG, MODELS, PROFILES
from ..coordinator import Coordinator, Listener

GAIN_TAPS = 3          # per mic area
SILENCE_S = 4.0
PROMPTS = {"wrist": "directly over mic_0 (wrist band)", "mid_forearm": "directly over mic_1 (5 cm toward the elbow)"}


def profile_path(name: str):
    return PROFILES / f"{name}.json"


def samples_dir(name: str):
    return PROFILES / f"{name}_samples"


def list_profiles() -> list[str]:
    PROFILES.mkdir(exist_ok=True)
    return sorted(p.stem for p in PROFILES.glob("*.json"))


def load_profile(name: str) -> dict | None:
    p = profile_path(name)
    return json.loads(p.read_text()) if p.exists() else None


def save_profile(prof: dict) -> None:
    PROFILES.mkdir(exist_ok=True)
    profile_path(prof["name"]).write_text(json.dumps(prof, indent=1))


def safe_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "-_")[:32] or "default"


def peak(ev, c: int) -> float:
    on = ev.onset
    return float(np.abs(ev.ch[c, max(on - 2, 0): on + int(0.05 * ev.rate)]).max())


class CalibrationApp(Listener):
    def __init__(self, coord: Coordinator, emit: Callable[[dict], None] | None = None,
                 samples_per_class: int = 5):
        self.coord = coord
        self.emit = emit or (lambda m: None)
        self.n_per = samples_per_class
        self.user = "default"
        self.step = "idle"
        self.message = "Press Start to begin calibration."
        self.progress = 0.0
        self.profile: dict = {}
        self.result: dict = {}
        self._live: list = []
        self._t_end = 0.0
        self._taps: list = []
        self._queue: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    # ----------------------------------------------------------- state
    def state(self) -> dict:
        self._check_silence()
        cur = self._queue[len(self._taps)] if self.step in ("gain", "baseline", "samples") and \
            len(self._taps) < len(self._queue) else None
        return dict(user=self.user, step=self.step, message=self.message, progress=round(self.progress, 3),
                    profile=self.profile, result=self.result, current=cur,
                    prompt=self._prompt(cur), taps_done=len(self._taps), taps_total=len(self._queue),
                    profiles=list_profiles())

    def _prompt(self, cur) -> str:
        if not cur:
            return ""
        m, z = cur
        if self.step == "gain":
            return f"Tap with a FINGER {PROMPTS[z]}"
        if self.step == "baseline":
            return "Tap lightly with a fingertip, then lift your finger clear of the arm"
        return f"Tap the {z.replace('_', ' ')} zone with: {m.upper()}"

    def _push(self) -> None:
        self.emit(dict(type="calib", **self.state()))

    # ----------------------------------------------------------- control
    def start(self, user: str) -> None:
        self.user = safe_name(user)
        self.coord.set_profile(None)               # calibrate on raw, uncorrected signals
        self.profile = dict(name=self.user, created=dt.datetime.now().isoformat(timespec="seconds"),
                            threshold=CFG["node"]["threshold"], refractory_ms=CFG["node"]["refractory_ms"],
                            gain_corr=[1.0, 1.0], baseline_db=None, noise_floor=None, samples=0,
                            adapted_model=None, simulated=self.coord.transport.is_simulated)
        self.result = {}
        self.step = "ready"
        self.progress = 0.0
        self.message = "Wizard started. Step 1: keep still and silent, then press Run."
        self._taps, self._queue = [], []
        self._push()

    def run_step(self, step: str) -> None:
        with self._lock:
            self.coord.sim_hint(None, None)
            self._taps = []
            self.progress = 0.0
            if step == "silence":
                self._live, self._t_end, self.step = [], time.time() + SILENCE_S, "silence"
                self.message = "Measuring noise floor - keep the arm still and quiet..."
            elif step == "gain":
                self._queue = [("finger", "wrist")] * GAIN_TAPS + [("finger", "mid_forearm")] * GAIN_TAPS
                self.step = "gain"
                self.message = "Gain check: tap with a finger exactly over each microphone."
                self._hint()
            elif step == "baseline":
                if not self.coord.node_config.get("probe_enabled", CFG["node"]["probe"]["enabled"]):
                    self.step, self.message = "baseline_skipped", "Probe is off - baseline skipped."
                else:
                    self._queue = [("none", "wrist")] * 3
                    self.step = "baseline"
                    self.message = "Probe baseline: nothing may touch the arm after each tap."
                    self._hint()
            elif step == "samples":
                self._queue = [(m, z) for z in CFG["zones"] for m in CFG["materials"] for _ in range(self.n_per)]
                self.step = "samples"
                self.message = "Personal samples: follow the on-screen prompt."
                self._hint()
            else:
                raise ValueError(step)
        self._push()

    def _hint(self) -> None:
        if len(self._taps) < len(self._queue):
            m, z = self._queue[len(self._taps)]
            self.coord.sim_hint(m, z)
        else:
            self.coord.sim_hint(None, None)

    # ----------------------------------------------------------- listener API
    def on_live(self, live: dict) -> None:
        if self.step == "silence":
            self._live.append(np.abs(live["samples"]).ravel())
            self.progress = min(1.0, 1 - (self._t_end - time.time()) / SILENCE_S)
            self._check_silence()

    def _check_silence(self) -> None:
        if self.step != "silence" or time.time() < self._t_end:
            return
        with self._lock:
            if self.step != "silence":
                return
            if not self._live:
                self.step, self.message = "ready", "No live data received - is the node started in LIVE mode?"
                return
            x = np.concatenate(self._live)
            # robust to a stray tap during the measurement: median-based noise sigma
            sigma = float(np.median(x)) / 1.4
            thr = int(np.clip(10.0 * sigma, 40, 1500))
            self.profile.update(threshold=thr, noise_floor=round(sigma, 2))
            self.coord.calibration_update(thr, self.profile["refractory_ms"])
            self.step, self.progress = "silence_done", 1.0
            self.message = f"Noise floor ~{self.profile['noise_floor']} counts rms -> tap threshold {thr}."
            self.result["silence"] = dict(noise_rms=self.profile["noise_floor"], threshold=thr)

    def on_data(self, ev) -> None:
        if self.step not in ("gain", "baseline", "samples") or ev.ch is None:
            return
        with self._lock:
            i = len(self._taps)
            if i >= len(self._queue):
                return
            m, z = self._queue[i]
            if self.step == "gain":
                self._taps.append((peak(ev, 0), peak(ev, 1)))
            elif self.step == "baseline":
                if ev.probe_db is None:
                    return
                self._taps.append(np.array(ev.probe_db))
            else:
                d = samples_dir(self.user) / m / z
                n = len(list(d.glob("*.npz"))) + 1 if d.exists() else 1
                D.save_event(d / f"{n}.npz", ev.raw, m, z, self.user,
                             "SIMULATED" if ev.simulated else "REAL")
                self._taps.append(1)
            self.progress = len(self._taps) / len(self._queue)
            if len(self._taps) >= len(self._queue):
                self._finish_step()
            else:
                self._hint()
        self._push()

    def _finish_step(self) -> None:
        self.coord.sim_hint(None, None)
        if self.step == "gain":
            n = GAIN_TAPS
            p = np.array(self._taps)
            p00, p01 = p[:n, 0].mean(), p[:n, 1].mean()      # tapped at mic_0: (ch0, ch1)
            p10, p11 = p[n:, 0].mean(), p[n:, 1].mean()      # tapped at mic_1
            g1_over_g0 = float(np.sqrt((p01 * p11) / max(p00 * p10, 1e-9)))   # attenuation cancels
            self.profile["gain_corr"] = [1.0, round(1.0 / g1_over_g0, 4)]
            self.result["gain"] = dict(mic1_over_mic0=round(g1_over_g0, 3), gain_corr=self.profile["gain_corr"])
            self.step, self.message = "gain_done", f"mic_1 is {g1_over_g0:.2f}x mic_0 -> correction {1 / g1_over_g0:.2f}"
        elif self.step == "baseline":
            base = np.mean(self._taps, axis=0)
            self.profile["baseline_db"] = base.tolist()
            self.result["baseline"] = dict(baseline_db=np.round(base, 1).tolist())
            self.step, self.message = "baseline_done", "Probe baseline stored."
        elif self.step == "samples":
            self.profile["samples"] = len(self._taps)
            self.step, self.message = "samples_done", f"{len(self._taps)} personal samples saved."
        self._save_and_activate()

    def _save_and_activate(self) -> None:
        save_profile(self.profile)
        self.activate(self.profile)

    def activate(self, prof: dict) -> None:
        """Make this profile the one used by F_PREPROCESS (+ its adapted model, if any)."""
        self.coord.set_profile(dict(prof))
        if prof.get("adapted_model"):
            from ..ml.model import ModelBundle
            mb = ModelBundle.load_tag(prof["adapted_model"])
            if mb:
                self.coord.set_models(mb)
        try:
            self.coord.calibration_update(int(prof["threshold"]), int(prof["refractory_ms"]))
        except Exception:
            pass

    # ----------------------------------------------------------- adaptation
    def adapt(self) -> None:
        if self.step == "adapting":
            return
        self.step, self.message, self.progress = "adapting", "Retraining with your samples (synthetic + yours)...", 0.1
        self._push()
        threading.Thread(target=self._adapt_job, daemon=True).start()

    def _adapt_job(self) -> None:
        try:
            from ..ml.model import ModelBundle
            from ..ml.train import featurize, train
            base = ModelBundle.latest()
            samples = D.load_dir(samples_dir(self.user))
            if len(samples) < 8 or base is None:
                self.step, self.message = "samples_done", "Need the personal-samples step (and a base model) first."
                self._push()
                return
            for s in samples:
                s["gain_corr"] = self.profile.get("gain_corr")
                s["baseline_db"] = self.profile.get("baseline_db")
            tr, te = samples[0::2], samples[1::2]
            ref = base.meta.get("ref_baseline_db")
            Xte, names = featurize(te, ref)
            yte = {k: np.array([s[k] for s in te]) for k in ("material", "zone")}

            def acc(bundle):
                out = {}
                for kind in ("material", "zone"):
                    pred = [bundle.predict(kind, x, names)[0] for x in Xte]
                    out[kind] = float(np.mean(np.array(pred) == yte[kind]))
                return out
            before = acc(base)
            self.progress = 0.3; self._push()
            tmp = train("synthetic", user_samples=tr, tag=f"tmp-adapt-{self.user}", set_latest=False,
                        verbose=False, evaluate=False)
            after = acc(ModelBundle.load_tag(tmp))
            import shutil
            shutil.rmtree(MODELS / tmp, ignore_errors=True)      # evaluation model is throw-away
            self.progress = 0.7; self._push()
            tag = f"adapt-{self.user}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            train("synthetic", user_samples=samples, tag=tag, set_latest=False, verbose=False, evaluate=False)
            self.profile["adapted_model"] = tag
            self.result["adapt"] = dict(before=before, after=after, n_train=len(tr), n_test=len(te), model=tag,
                                        note="evaluated on held-out half of YOUR samples; final model uses all")
            self._save_and_activate()
            self.step = "done"
            self.message = (f"Adapted. Material accuracy on your held-out taps: "
                            f"{before['material'] * 100:.0f}% -> {after['material'] * 100:.0f}%")
            self.progress = 1.0
        except Exception as e:  # surface, don't crash the thread silently
            self.step, self.message = "samples_done", f"Adaptation failed: {e!r}"
        self._push()
