"""Physics-inspired synthetic tap generator. EVERYTHING produced here is SYNTHETIC.

Model
 * A tap excites a few damped modes whose frequencies / damping depend on the object.
 * Waves travel along the arm: delay = distance / speed, amplitude falls with distance,
   faster for high frequencies. Mic distances depend on the tap zone -> TDOA + amplitude
   ratio carry the location.
 * The chirp probe passes through a bank of arm resonances; touching objects change the
   resonance frequencies / Q / gain.
 * Variation: tap strength, per-user skin + gain mismatch, noise, hum, crosstalk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from ..config import CFG, window_sizes

# (freq Hz, relative amplitude) modes, decay time constant (s), attack (s)
TAP_MODEL = {
    "finger":  dict(modes=[(70, 1.0), (140, 0.7), (320, 0.3)], tau=0.007, attack=0.0025, click=0.15),
    "metal":   dict(modes=[(1500, 0.9), (2650, 1.0), (3900, 0.7)], tau=0.075, attack=0.0003, click=0.5),
    "wood":    dict(modes=[(320, 1.0), (720, 0.7), (1250, 0.4)], tau=0.012, attack=0.0008, click=0.3),
    "plastic": dict(modes=[(900, 0.9), (1750, 1.0), (2800, 0.6)], tau=0.010, attack=0.0004, click=0.4),
}
# probe resonances of bare skin (f, Q, gain)
PROBE_BASE = [(600, 6.0, 1.0), (1700, 8.0, 0.8), (3000, 10.0, 0.6)]
PROBE_MODEL = {  # freq scale, Q scale, gain scale, extra resonances
    "none":    dict(f=1.0, q=1.0, g=1.0, extra=[]),
    "finger":  dict(f=0.9, q=0.35, g=0.7, extra=[]),
    "metal":   dict(f=1.15, q=1.5, g=1.1, extra=[(2400, 25.0, 0.9)]),
    "wood":    dict(f=0.97, q=0.8, g=0.9, extra=[(1200, 12.0, 0.5)]),
    "plastic": dict(f=1.03, q=0.9, g=0.95, extra=[(2000, 10.0, 0.4)]),
}
# tap distance (cm) to (mic_0, mic_1)
ZONE_GEOM = {"wrist": (1.5, 6.5), "mid_forearm": (7.5, 2.5)}
SPEED = 6000.0  # cm/s  (60 m/s surface wave)


@dataclass
class VirtualUser:
    v_scale: float = 1.0      # skin wave speed factor
    damp_scale: float = 1.0
    gain1: float = 1.0        # mic_1 gain mismatch
    noise: float = 5.0        # counts rms

    @staticmethod
    def random(rng) -> "VirtualUser":
        return VirtualUser(rng.uniform(0.8, 1.25), rng.uniform(0.8, 1.3), rng.uniform(0.8, 1.25),
                           rng.uniform(4.0, 14.0))


def _bandpass(f0: float, q: float, fs: int):
    w0 = 2 * np.pi * min(f0, 0.45 * fs) / fs
    alpha = np.sin(w0) / (2 * q)
    b = np.array([alpha, 0, -alpha])
    a = np.array([1 + alpha, -2 * np.cos(w0), 1 - alpha])
    return b / a[0], a / a[0]


class EventSynth:
    def __init__(self, cfg: dict | None = None, seed: int = 0, rate: int | None = None):
        self.cfg = cfg or CFG
        self.rng = np.random.default_rng(seed)
        self.w = window_sizes(self.cfg, rate)
        self.fs = self.w["rate"]
        self.probe_cfg = self.cfg["node"]["probe"]
        self.n_pt = self.w["n_pre"] + self.w["n_tap"]

    # ------------------------------------------------------------------ pieces
    def _tap_at(self, d_cm: float, user: VirtualUser, params: dict) -> np.ndarray:
        n = self.n_pt
        t = (np.arange(n) - self.w["n_pre"]) / self.fs
        u = t - d_cm / (SPEED * user.v_scale)
        m = u > 0
        uu = np.where(m, u, 0.0)
        tau = params["tau"] / user.damp_scale
        sig = np.zeros(n)
        for f, a in params["modes"]:
            lam = 9.0 / (1.0 + f / 1500.0)               # cm, shorter for high f
            amp = a * np.exp(-d_cm / lam)
            sig += amp * np.sin(2 * np.pi * f * uu + params["phase"][f]) * np.exp(-uu / tau)
        sig *= (1 - np.exp(-uu / params["attack"])) * m
        click = params["click_noise"] * np.exp(-uu / 0.0015) * m * np.exp(-d_cm / 6.0)
        return sig + params["click"] * click

    def _probe(self, material: str, zone: str, user: VirtualUser) -> np.ndarray:
        p, fs = self.probe_cfg, self.fs
        T = p["duration_ms"] / 1000
        n_total = int(round((p["duration_ms"] + p["tail_ms"]) * fs / 1000))
        t = np.arange(n_total) / fs
        k = (p["f_end"] - p["f_start"]) / T
        drive = np.where(t < T, np.sin(2 * np.pi * (p["f_start"] * t + 0.5 * k * t * t)), 0.0)
        ramp = np.clip(t / 0.003, 0, 1) * np.clip((T - t) / 0.003, 0, 1)
        drive = drive * np.where(t < T, ramp, 0.0)
        pm = PROBE_MODEL[material]
        zshift = 1.0 if zone == "wrist" else 0.94
        res = [(f * pm["f"] * zshift, q * pm["q"] / user.damp_scale, g * pm["g"]) for f, q, g in PROBE_BASE]
        res += [(f * zshift, q, g) for f, q, g in pm["extra"]]
        out = np.zeros((2, n_total))
        for c in range(2):
            y = np.zeros(n_total)
            for f, q, g in res:
                b, a = _bandpass(f, q, fs)
                y += g * lfilter(b, a, drive) * (1.0 + 0.1 * self.rng.standard_normal())
            out[c] = y * (260.0 if c == 0 else 260.0 * user.gain1)
        skip = int(round(p["skip_ms"] * fs / 1000))
        return out[:, skip:]

    def _noise(self, n: int, user: VirtualUser) -> np.ndarray:
        t = np.arange(n) / self.fs
        out = user.noise * self.rng.standard_normal((2, n))
        out += 3.0 * np.sin(2 * np.pi * 50 * t + self.rng.uniform(0, 6.28))   # mains hum
        out += self.rng.normal(0, 3) * np.linspace(-1, 1, n)                  # slow wander
        return out

    def idle(self, n: int, user: VirtualUser | None = None) -> np.ndarray:
        return self._noise(n, user or VirtualUser())

    # ------------------------------------------------------------------ public
    def synth(self, material: str, zone: str, user: VirtualUser | None = None,
              probe: bool = True, strength: float | None = None,
              probe_material: str | None = None) -> dict:
        """One event in the node's DATA format (dict like protocol.dec_data)."""
        rng = self.rng
        user = user or VirtualUser()
        strength = strength if strength is not None else rng.uniform(150.0, 1100.0)
        params = dict(TAP_MODEL[material])
        params["tau"] = params["tau"] * rng.uniform(0.6, 1.6)
        params["modes"] = [(f * rng.uniform(0.85, 1.15), a * rng.uniform(0.5, 1.5)) for f, a in params["modes"]]
        params["phase"] = {f: rng.uniform(0, 2 * np.pi) for f, _ in params["modes"]}
        params["click_noise"] = np.convolve(rng.standard_normal(self.n_pt), np.ones(3) / 3, mode="same")
        d0, d1 = ZONE_GEOM[zone]
        d0 = max(d0 + rng.normal(0, 0.5), 0.5)
        d1 = max(d1 + rng.normal(0, 0.5), 0.5)
        ch = np.zeros((2, self.n_pt))
        ch[0] = self._tap_at(d0, user, params)
        ch[1] = self._tap_at(d1, user, params) * user.gain1
        ch *= strength / max(np.abs(ch).max(), 1e-9)          # louder channel peaks at `strength`
        ch = ch + rng.uniform(0.02, 0.07) * ch[::-1]          # crosstalk
        ch += self._noise(self.n_pt, user)
        ch = np.clip(ch, -2047, 2047)
        pr = None
        if probe:
            pr = self._probe(probe_material or material, zone, user)
            pr = pr + self._noise(pr.shape[1], user) + 0.04 * pr[::-1]
            pr = np.clip(pr, -2047, 2047).astype(np.float32)
        return dict(ch=ch.astype(np.float32), probe=pr, rate_hz=self.fs, n_pre=self.w["n_pre"],
                    n_tap=self.w["n_tap"], probe_used=probe, simulated=True,
                    material=material, zone=zone)

    def baseline_probe(self, user: VirtualUser | None = None) -> np.ndarray:
        """Probe response with nothing touching (used by the calibration baseline step)."""
        user = user or VirtualUser()
        pr = self._probe("none", "wrist", user)
        return np.clip(pr + self._noise(pr.shape[1], user), -2047, 2047).astype(np.float32)
