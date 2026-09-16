"""Cutting the analysis window and 2-minute binning."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cut_window(t: np.ndarray, f: np.ndarray, lo: float = -20.0, hi: float = 90.0):
    """Keep samples with lo <= t <= hi (minutes from the flare peak)."""
    m = (t >= lo) & (t <= hi)
    return t[m], f[m]


def to_2min(t: np.ndarray, f: np.ndarray, phase: float, min_n: int = 20):
    """Average to 2-minute bins with the given start phase (0-2 min).

    Bins with fewer than ``min_n`` raw points are dropped (about 60 are expected).
    """
    b = np.floor((t + phase) / 2.0).astype(int)
    g_ = pd.DataFrame({"b": b, "t": t, "f": f}).groupby("b").agg(
        t=("t", "mean"), f=("f", "mean"), n=("f", "size"))
    g_ = g_[g_.n >= min_n]
    return g_.t.values, g_.f.values


def stellar_window(time, flux, tpeak: float, lo: float = -20.0, hi: float = 90.0):
    """Convert BTJD days to minutes from the peak and cut the window (no re-binning)."""
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    tm = (time - float(tpeak)) * 1440.0
    ok = np.isfinite(tm) & np.isfinite(flux)
    tm, flux = tm[ok], flux[ok]
    order = np.argsort(tm)
    return cut_window(tm[order], flux[order], lo, hi)
