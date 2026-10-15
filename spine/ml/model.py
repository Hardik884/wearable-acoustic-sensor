"""Trained model bundle: material + zone classifiers, loaded from models/<tag>/."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from ..config import CFG, MODELS


def feature_subset(names: list[str], kind: str, variant: str = "combined") -> list[str]:
    if kind == "zone":
        keep = [n for n in names if n.startswith("xc_") or n == "probe_ratio_db"
                or any(n.endswith(s) for s in ("_rms", "_peak", "_centroid"))]
        return keep
    if variant == "tap":
        return [n for n in names if n.startswith("tap_")]
    if variant == "probe":
        return [n for n in names if n.startswith("probe_")]
    return [n for n in names if n.startswith(("tap_", "probe_"))]


class ModelBundle:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.models = {k: joblib.load(self.path / f"{k}.joblib") for k in ("material", "zone")
                       if (self.path / f"{k}.joblib").exists()}
        self.unknown_threshold = CFG["ml"]["unknown_threshold"]

    @property
    def tag(self) -> str:
        return self.meta["tag"]

    def predict(self, kind: str, feats: np.ndarray, names: list[str]):
        m = self.models.get(kind)
        if m is None:
            return "unknown", 0.0, {}
        idx = [names.index(n) for n in self.meta["features"][kind]]
        x = np.asarray(feats, dtype=float)[idx][None, :]
        p = m.predict_proba(x)[0]
        classes = list(m.classes_)
        k = int(np.argmax(p))
        return classes[k], float(p[k]), {c: float(v) for c, v in zip(classes, p)}

    @staticmethod
    def latest() -> "ModelBundle | None":
        ptr = MODELS / "latest.json"
        if not ptr.exists():
            return None
        tag = json.loads(ptr.read_text())["tag"]
        d = MODELS / tag
        return ModelBundle(d) if (d / "meta.json").exists() else None

    @staticmethod
    def load_tag(tag: str) -> "ModelBundle | None":
        d = MODELS / tag
        return ModelBundle(d) if (d / "meta.json").exists() else None
