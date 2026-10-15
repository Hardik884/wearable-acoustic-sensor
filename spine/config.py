"""Loads config.yaml once; exposes project paths."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SYNTH_DIR = DATA / "synthetic"
REAL_DIR = DATA / "real"
SIMREC_DIR = DATA / "simulated_recordings"
MODELS = ROOT / "models"
PROFILES = ROOT / "profiles"
DB_PATH = DATA / "events.db"
RULES_PATH = ROOT / "safety_rules.yaml"


def load(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load()


def window_sizes(cfg: dict | None = None, rate: int | None = None) -> dict:
    n = (cfg or CFG)["node"]
    fs = rate or n["sample_rate"]
    ms = lambda v: int(round(fs * v / 1000))
    p = n["probe"]
    return dict(rate=fs, n_pre=ms(n["pre_ms"]), n_tap=ms(n["tap_ms"]),
                n_probe=ms(p["duration_ms"] + p["tail_ms"] - p["skip_ms"]))
