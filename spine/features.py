"""Signal-processing building blocks: onset, spectral features, MFCC, GCC-PHAT, probe bands."""
from __future__ import annotations

import numpy as np
from scipy.fft import dct, rfft, irfft

PROBE_BANDS = [(300, 800), (800, 1500), (1500, 2500), (2500, 4000)]
N_MFCC = 8
SEG_MS = 50.0           # analysis segment after onset
GCC_MS = 20.0           # GCC-PHAT segment
MAX_LAG_MS = 1.6        # physically plausible TDOA range


def find_onset(x: np.ndarray, n_pre: int, k_noise: float = 5.0, frac: float = 0.12) -> int:
    """First sample whose |x| exceeds max(k*noise, frac*peak). x: (n_ch, n)."""
    a = np.abs(x).max(axis=0)
    pre = x[:, : max(n_pre - 2, 8)]
    noise = float(np.std(pre))
    thr = max(k_noise * noise, frac * a.max(), 8.0)
    idx = np.nonzero(a > thr)[0]
    return int(idx[0]) if idx.size else n_pre


def _mel_fb(n_fft: int, fs: int, n_mel: int = 20, fmin: float = 50.0) -> np.ndarray:
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fs / 2), n_mel + 2))
    bins = np.floor((n_fft + 1) * pts / fs).astype(int)
    fb = np.zeros((n_mel, n_fft // 2 + 1))
    for m in range(1, n_mel + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        c = max(c, l + 1); r = max(r, c + 1)
        fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


_FB_CACHE: dict = {}


def mfcc(seg: np.ndarray, fs: int, n: int = N_MFCC) -> np.ndarray:
    n_fft = 1024
    key = (n_fft, fs)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_fb(n_fft, fs)
    w = seg[:n_fft] * np.hanning(min(len(seg), n_fft))
    spec = np.abs(rfft(w, n_fft)) ** 2
    mel = np.log(_FB_CACHE[key] @ spec + 1e-6)
    return dct(mel, type=2, norm="ortho")[1 : n + 1]


def spectral(seg: np.ndarray, fs: int) -> tuple[float, float]:
    """(centroid Hz, 85% rolloff Hz)."""
    n_fft = 1024
    mag = np.abs(rfft(seg[:n_fft] * np.hanning(min(len(seg), n_fft)), n_fft))
    f = np.fft.rfftfreq(n_fft, 1 / fs)
    tot = mag.sum() + 1e-9
    cent = float((f * mag).sum() / tot)
    c = np.cumsum(mag)
    roll = float(f[min(np.searchsorted(c, 0.85 * c[-1]), len(f) - 1)])
    return cent, roll


def envelope(x: np.ndarray, fs: int) -> np.ndarray:
    k = max(3, int(fs * 0.001))
    return np.convolve(np.abs(x), np.ones(k) / k, mode="same")


def attack_decay(x: np.ndarray, fs: int) -> tuple[float, float]:
    """x starts at onset. attack: time to peak (ms). decay: time from peak to 20% (ms, capped)."""
    env = envelope(x, fs)
    pk = int(np.argmax(env[: int(0.03 * fs)] if len(env) > int(0.03 * fs) else env))
    peak = env[pk] + 1e-9
    tail = env[pk:]
    below = np.nonzero(tail < 0.2 * peak)[0]
    dec = (below[0] if below.size else len(tail)) / fs * 1000
    return pk / fs * 1000, float(dec)


def gcc_phat(a: np.ndarray, b: np.ndarray, fs: int, max_lag_ms: float = MAX_LAG_MS) -> tuple[float, float]:
    """Delay (ms) of b relative to a (positive: b lags a) and peak sharpness 0..1."""
    n = len(a) + len(b)
    X = rfft(a, n); Y = rfft(b, n)
    R = np.conj(X) * Y
    R /= np.abs(R) + 1e-9
    cc = irfft(R, n)
    L = int(max_lag_ms * fs / 1000)
    cc = np.concatenate([cc[-L:], cc[: L + 1]])
    k = int(np.argmax(cc))
    frac = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    lag = (k - L) + frac
    return lag / fs * 1000, float(cc[k])


def probe_bands_db(probe: np.ndarray, rate: int) -> np.ndarray:
    """(n_ch, 4) band energies in dB of the probe-response window."""
    n_fft = 2048
    f = np.fft.rfftfreq(n_fft, 1 / rate)
    out = np.zeros((probe.shape[0], len(PROBE_BANDS)))
    for c in range(probe.shape[0]):
        x = probe[c] - probe[c].mean()
        p = np.abs(rfft(x * np.hanning(len(x)), n_fft)) ** 2 / len(x)
        for k, (lo, hi) in enumerate(PROBE_BANDS):
            m = (f >= lo) & (f < min(hi, rate / 2))
            out[c, k] = 10 * np.log10(p[m].sum() + 1.0)
    return out
