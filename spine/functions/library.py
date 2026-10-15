"""SPINE function library: pluggable processing blocks, each = code + parameters.

Pipeline per tap:  F_PREPROCESS -> F_FEATURES -> F_LOCALIZE -> F_CLASSIFY -> F_FUSION
Functions are registered by code, so a pipeline is just a list of {code, params} dicts and
can be rebuilt/extended (or moved onto the node in the next phase) without touching apps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import features as F

REGISTRY: dict[str, type] = {}


def register(cls):
    REGISTRY[cls.code] = cls
    return cls


@dataclass
class Env:
    """Shared, swappable context for functions: active user profile + trained models."""
    profile: dict | None = None        # gain_corr, baseline_db, ...
    models: Any = None                 # spine.ml.model.ModelBundle or None


@dataclass
class TapEvent:
    raw: dict                          # decoded DATA payload (ch, probe, rate_hz, n_pre, ...)
    ts: float = 0.0
    node_id: int = 0
    env: Env = field(default_factory=Env)
    # filled by functions
    ch: np.ndarray | None = None
    probe: np.ndarray | None = None
    onset: int = 0
    probe_db: np.ndarray | None = None        # (2, 4) baseline-normalised
    feats: np.ndarray | None = None
    names: list[str] = field(default_factory=list)
    zone: str = "unknown"
    zone_conf: float = 0.0
    zone_probs: dict = field(default_factory=dict)
    material: str = "unknown"
    material_conf: float = 0.0
    material_probs: dict = field(default_factory=dict)
    final_material: str = "unknown"
    final_zone: str = "unknown"
    confidence: float = 0.0

    @property
    def rate(self) -> int:
        return int(self.raw["rate_hz"])

    @property
    def simulated(self) -> bool:
        return bool(self.raw.get("simulated"))

    def to_dict(self) -> dict:
        return dict(ts=self.ts, node=self.node_id, zone=self.final_zone, material=self.final_material,
                    zone_conf=round(self.zone_conf, 3), material_conf=round(self.material_conf, 3),
                    zone_probs={k: round(v, 3) for k, v in self.zone_probs.items()},
                    material_probs={k: round(v, 3) for k, v in self.material_probs.items()},
                    probe_used=bool(self.raw.get("probe_used")), simulated=self.simulated,
                    tdoa_ms=self._f("xc_tdoa_ms"), ratio_db=self._f("xc_ratio_db"))

    def _f(self, name: str):
        if self.feats is not None and name in self.names:
            return round(float(self.feats[self.names.index(name)]), 3)
        return None


class SpineFunction:
    code = "F_BASE"

    def __init__(self, **params):
        self.params = params

    def __call__(self, ev: TapEvent) -> TapEvent:  # pragma: no cover
        raise NotImplementedError


@register
class Preprocess(SpineFunction):
    """DC removal, per-user gain correction, onset detection, probe baseline normalisation."""
    code = "F_PREPROCESS"

    def __call__(self, ev: TapEvent) -> TapEvent:
        raw = ev.raw
        ch = np.array(raw["ch"], dtype=np.float64)
        n_pre = int(raw["n_pre"])
        prof = ev.env.profile or {}
        ch -= np.median(ch[:, : max(n_pre - 2, 8)], axis=1, keepdims=True)
        gc = prof.get("gain_corr")
        if gc:
            ch *= np.asarray(gc, dtype=float)[:, None]
        ev.ch = ch
        ev.onset = F.find_onset(ch, n_pre)
        ev.probe = None
        ev.probe_db = None
        if raw.get("probe") is not None:
            pr = np.array(raw["probe"], dtype=np.float64)
            if gc:
                pr *= np.asarray(gc, dtype=float)[:, None]
            ev.probe = pr
            db = F.probe_bands_db(pr, ev.rate)
            base = prof.get("baseline_db")
            ref = self.params.get("ref_baseline_db")
            if ref is None and ev.env.models is not None:
                ref = ev.env.models.meta.get("ref_baseline_db")
            if base is not None and ref is not None:
                db = db - np.asarray(base) + np.asarray(ref)
            ev.probe_db = db
        return ev


@register
class Features(SpineFunction):
    """Tap + cross-channel + probe features. Output vector order is stable (ev.names)."""
    code = "F_FEATURES"

    def __call__(self, ev: TapEvent) -> TapEvent:
        fs, ch, on = ev.rate, ev.ch, ev.onset
        n_seg = int(F.SEG_MS * fs / 1000)
        names: list[str] = []
        vals: list[float] = []

        def add(n, v):
            names.append(n); vals.append(float(v))

        seg = ch[:, max(on - 2, 0): max(on - 2, 0) + n_seg]
        for c in range(2):
            x = seg[c]
            rms = np.sqrt(np.mean(x ** 2))
            pk = np.abs(x).max()
            xn = x / (pk + 1e-9)
            cent, roll = F.spectral(xn, fs)
            att, dec = F.attack_decay(x, fs)
            add(f"tap_c{c}_rms", np.log10(1 + rms))
            add(f"tap_c{c}_peak", np.log10(1 + pk))
            add(f"tap_c{c}_centroid", cent)
            add(f"tap_c{c}_rolloff", roll)
            add(f"tap_c{c}_attack_ms", att)
            add(f"tap_c{c}_decay_ms", dec)
            for k, m in enumerate(F.mfcc(xn, fs), 1):
                add(f"tap_c{c}_mfcc{k}", m)
        # cross-channel (localisation)
        r0 = np.sqrt(np.mean(seg[0] ** 2)) + 1.0
        r1 = np.sqrt(np.mean(seg[1] ** 2)) + 1.0
        add("xc_ratio_db", 20 * np.log10(r0 / r1))
        add("xc_peak_ratio_db", 20 * np.log10((np.abs(seg[0]).max() + 1) / (np.abs(seg[1]).max() + 1)))
        g0 = max(on - 4, 0)
        n_g = int(F.GCC_MS * fs / 1000)
        tdoa, sharp = F.gcc_phat(ch[0, g0: g0 + n_g], ch[1, g0: g0 + n_g], fs)
        add("xc_tdoa_ms", tdoa)
        add("xc_gcc_peak", sharp)
        # per-channel onset difference, ms (positive: mic_1 later)
        def th(x):
            idx = np.nonzero(np.abs(x) > max(0.2 * np.abs(x).max(), 8.0))[0]
            return int(idx[0]) if idx.size else on
        add("xc_onset_diff_ms", (th(ch[1]) - th(ch[0])) / fs * 1000)
        # probe-response band energies (NaN if probe absent -> imputed by the model)
        for c in range(2):
            for k in range(len(F.PROBE_BANDS)):
                add(f"probe_c{c}_b{k}", ev.probe_db[c, k] if ev.probe_db is not None else np.nan)
        add("probe_ratio_db", (ev.probe_db[0].sum() - ev.probe_db[1].sum()) if ev.probe_db is not None else np.nan)
        ev.feats, ev.names = np.array(vals), names
        return ev


def _predict(models, kind: str, feats: np.ndarray, names: list[str]):
    return models.predict(kind, feats, names) if models is not None else ("unknown", 0.0, {})


@register
class Localize(SpineFunction):
    """Zone (where on the forearm)."""
    code = "F_LOCALIZE"

    def __call__(self, ev: TapEvent) -> TapEvent:
        ev.zone, ev.zone_conf, ev.zone_probs = _predict(ev.env.models, "zone", ev.feats, ev.names)
        return ev


@register
class Classify(SpineFunction):
    """Material (what object)."""
    code = "F_CLASSIFY"

    def __call__(self, ev: TapEvent) -> TapEvent:
        ev.material, ev.material_conf, ev.material_probs = _predict(ev.env.models, "material", ev.feats, ev.names)
        return ev


@register
class Fusion(SpineFunction):
    """Combine localisation + classification into the final decision (unknown on low confidence)."""
    code = "F_FUSION"

    def __call__(self, ev: TapEvent) -> TapEvent:
        thr_m = self.params.get("unknown_threshold", 0.5)
        thr_z = self.params.get("zone_threshold", 0.5)
        ev.final_material = ev.material if ev.material_conf >= thr_m else "unknown"
        ev.final_zone = ev.zone if ev.zone_conf >= thr_z else "unknown"
        ev.confidence = float(np.sqrt(max(ev.material_conf, 0) * max(ev.zone_conf, 0)))
        return ev


DEFAULT_PIPELINE = [
    {"code": "F_PREPROCESS"}, {"code": "F_FEATURES"}, {"code": "F_LOCALIZE"},
    {"code": "F_CLASSIFY"}, {"code": "F_FUSION"},
]


def build_pipeline(spec: list[dict] | None = None, **extra) -> list[SpineFunction]:
    out = []
    for item in spec or DEFAULT_PIPELINE:
        params = {**extra, **item.get("params", {})}
        out.append(REGISTRY[item["code"]](**params))
    return out


def run_pipeline(pipeline: list[SpineFunction], ev: TapEvent) -> TapEvent:
    for fn in pipeline:
        ev = fn(ev)
    return ev
