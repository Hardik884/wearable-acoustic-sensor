"""Datasets: synthetic (one npz), real recordings (one npz per tap), personal samples.

A "sample" is a dict: ch, probe (or None), rate_hz, n_pre, material, zone, user, source,
baseline_db (or None). source is one of SYNTHETIC / SIMULATED / REAL.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import CFG, REAL_DIR, SYNTH_DIR
from .sim.generator import EventSynth, VirtualUser

SYNTH_FILE = SYNTH_DIR / "SYNTHETIC_dataset.npz"


def generate_synthetic(per_class: int | None = None, seed: int | None = None, n_users: int = 8,
                       cfg: dict | None = None, path: Path | None = None, verbose: bool = True) -> Path:
    cfg = cfg or CFG
    per_class = per_class or cfg["synthetic"]["per_class"]
    seed = cfg["synthetic"]["seed"] if seed is None else seed
    synth = EventSynth(cfg, seed=seed)
    users = [VirtualUser.random(synth.rng) for _ in range(n_users)]
    ch, pr, mat, zon, usr = [], [], [], [], []
    t0 = time.time()
    for m in cfg["materials"]:
        for z in cfg["zones"]:
            for i in range(per_class):
                u = int(synth.rng.integers(n_users))
                ev = synth.synth(m, z, users[u])
                ch.append(ev["ch"].astype(np.int16)); pr.append(ev["probe"].astype(np.int16))
                mat.append(m); zon.append(z); usr.append(u)
        if verbose:
            print(f"  synthetic {m}: done ({time.time() - t0:.1f}s)")
    path = path or SYNTH_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ch=np.stack(ch), probe=np.stack(pr), material=np.array(mat),
                        zone=np.array(zon), user=np.array(usr), rate=synth.fs, n_pre=synth.w["n_pre"],
                        label="SYNTHETIC")
    if verbose:
        print(f"wrote {path} ({len(mat)} events) -- SYNTHETIC DATA")
    return path


def load_synthetic(path: Path | None = None) -> list[dict]:
    path = path or SYNTH_FILE
    if not path.exists():
        generate_synthetic(path=path)
    d = np.load(path)
    rate, n_pre = int(d["rate"]), int(d["n_pre"])
    ch, probe = d["ch"].astype(np.float32), d["probe"].astype(np.float32)   # decompress once
    mat, zon, usr = d["material"], d["zone"], d["user"]
    return [dict(ch=ch[i], probe=probe[i], rate_hz=rate, n_pre=n_pre, material=str(mat[i]),
                 zone=str(zon[i]), user=f"synth{int(usr[i])}", source="SYNTHETIC", probe_used=True,
                 baseline_db=None)
            for i in range(len(mat))]


def save_event(path: Path, ev: dict, material: str, zone: str, user: str, source: str,
               baseline_db=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, ch=np.asarray(ev["ch"], dtype=np.float32),
        probe=np.asarray(ev["probe"], dtype=np.float32) if ev.get("probe") is not None else np.zeros((0,)),
        rate_hz=int(ev["rate_hz"]), n_pre=int(ev["n_pre"]), material=material, zone=zone, user=user,
        source=source, baseline_db=np.asarray(baseline_db, dtype=float) if baseline_db is not None else np.zeros((0,)),
        ts=time.time())


def load_event(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    probe = d["probe"] if d["probe"].ndim == 2 else None
    base = d["baseline_db"] if d["baseline_db"].size else None
    return dict(ch=d["ch"], probe=probe, rate_hz=int(d["rate_hz"]), n_pre=int(d["n_pre"]),
                material=str(d["material"]), zone=str(d["zone"]), user=str(d["user"]),
                source=str(d["source"]), probe_used=probe is not None, baseline_db=base)


def load_dir(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.glob("*/*/*.npz")):
        try:
            out.append(load_event(p))
        except Exception as e:  # corrupt file should never kill training
            print("skipping", p, e)
    return out


def load_real(root: Path | None = None) -> list[dict]:
    return load_dir(root or REAL_DIR)


def count_dir(root: Path, materials: list[str], zones: list[str]) -> dict:
    return {m: {z: len(list((root / m / z).glob("*.npz"))) for z in zones} for m in materials}
