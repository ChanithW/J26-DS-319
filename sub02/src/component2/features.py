"""Flare shape features and preprocessing (ported exactly from the research notebook)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

trap = getattr(np, "trapezoid", None) or np.trapz

FEATS13 = ["rise10_min", "rise50_min", "decay_tau_min", "fwhm_min", "t50_min", "t25_min",
           "g5", "g10", "asym", "rise_slope", "decay_slope", "eff_dur_min", "rebright"]
TIMEF = ["rise10_min", "rise50_min", "decay_tau_min", "fwhm_min", "t50_min", "t25_min",
         "t10_min", "eff_dur_min"]


def shape_feats(tm: np.ndarray, f: np.ndarray) -> dict | None:
    """Compute flare shape features on a 2-minute curve; None if the flare is unusable.

    ``tm`` is minutes relative to the catalogue peak, ``f`` the flux.
    """
    tm = np.asarray(tm, float)
    f = np.asarray(f, float)
    if len(tm) == 0 or len(tm) != len(f):
        return None
    bgm = tm < -8
    if bgm.sum() < 3:
        return None
    bg = np.median(f[bgm])
    near = np.where(np.abs(tm) <= 6)[0]
    if len(near) == 0:
        return None
    ip = near[np.argmax(f[near])]
    amp = f[ip] - bg
    if amp <= 0 or bg <= 0:
        return None
    g = (f - bg) / amp                                   # peak = 1, background = 0
    post = g[ip:]
    cut = len(post)
    mn = post[0]
    for i in range(1, len(post)):                        # re-brightening cut-off
        mn = min(mn, post[i])
        if post[i] > max(mn, 0.0) * 1.5 and post[i] > 0.15:
            cut = i
            break
    tt, gg = tm[ip:ip + cut] - tm[ip], g[ip:ip + cut]
    pre_t, pre_g = tm[:ip + 1] - tm[ip], g[:ip + 1]
    a10, a50 = np.where(pre_g >= 0.1)[0], np.where(pre_g >= 0.5)[0]
    rise10 = -pre_t[a10[0]] if len(a10) else np.nan
    rise50 = -pre_t[a50[0]] if len(a50) else np.nan

    def t_below(level):                                  # first crossing below `level`, linear interpolation
        idx = np.where(gg < level)[0]
        if len(idx) == 0 or idx[0] == 0:
            return np.nan
        i = idx[0]
        x0, x1, y0, y1 = tt[i - 1], tt[i], gg[i - 1], gg[i]
        return x0 + (y0 - level) * (x1 - x0) / (y0 - y1) if y0 != y1 else x1

    t50, t25, t10 = t_below(0.5), t_below(0.25), t_below(0.1)

    tau = np.nan                                         # exponential decay time: scaled, multi-start fit
    ok = gg > 0
    if ok.sum() >= 5:
        best = None
        for tau0 in (3, 5, 10, 20, 40):
            try:
                popt, _ = curve_fit(lambda t, A, s: A * np.exp(-t / s), tt[ok], gg[ok], p0=[1, tau0],
                                    bounds=([0, 0.5], [5, 60]), maxfev=5000)
                sse = np.sum((gg[ok] - popt[0] * np.exp(-tt[ok] / popt[1])) ** 2)
                if best is None or sse < best[0]:
                    best = (sse, popt[1])
            except Exception:
                pass
        if best is not None and 0.51 < best[1] < 59.9:
            tau = best[1]                                # NaN if the fit hit its limits

    at = lambda m: float(np.interp(m, tt, gg)) if len(tt) > 1 and m <= tt[-1] else np.nan
    dts = np.diff(tm)
    rs = np.max(np.diff(g[:ip + 1]) / dts[:ip]) if ip >= 1 else np.nan
    ds = np.min(np.diff(g[ip:ip + cut]) / dts[ip:ip + cut - 1]) if cut >= 2 else np.nan
    return dict(rise10_min=rise10, rise50_min=rise50, decay_tau_min=tau, fwhm_min=rise50 + t50,
                t50_min=t50, t25_min=t25, t10_min=t10, g5=at(5), g10=at(10), g20=at(20), g40=at(40),
                asym=(t50 + 1) / (rise50 + 1), rise_slope=rs, decay_slope=ds,
                eff_dur_min=float(trap(np.clip(gg, 0, None), tt)), rebright=int(cut < len(post)))


def prep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Log-transform the skewed time features (no filling or scaling)."""
    X = df[cols].copy()
    for c in cols:
        if c in TIMEF:
            X[c] = np.log1p(X[c].clip(lower=0))
    return X


def plot_curve(tm: np.ndarray, f: np.ndarray) -> dict | None:
    """Normalised curve for plotting: (flux - background) / amplitude, peak = 1.

    Uses the same background and peak rule as ``shape_feats``; None if unusable.
    """
    tm = np.asarray(tm, float)
    f = np.asarray(f, float)
    bgm = tm < -8
    near = np.where(np.abs(tm) <= 6)[0]
    if len(tm) == 0 or bgm.sum() < 3 or len(near) == 0:
        return None
    bg = np.median(f[bgm])
    ip = near[np.argmax(f[near])]
    amp = f[ip] - bg
    if amp <= 0 or bg <= 0:
        return None
    g = (f - bg) / amp
    return {"t_min": [round(float(x), 3) for x in tm], "g": [round(float(x), 5) for x in g]}
