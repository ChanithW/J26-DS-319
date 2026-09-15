"""Run the model over a whole TESS light curve and return one row per flare."""

import time as _time

import numpy as np
import pandas as pd

from .energy import TIER_NAMES
from .features import flare_snr, make_channels, sliding_windows

EVENT_COLUMNS = ["peak_time_btjd", "flare_prob", "energy_tier", "tier_conf", "amplitude", "snr", "weak"]


def download_light_curve(tic, sector=None):
    """Download a 2-minute SPOC light curve with lightkurve (needs internet).

    sector=None takes the first available sector.
    Returns (time_btjd, normalised_flux, sector_used, all_2min_sectors).
    """
    import lightkurve as lk

    res = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC", exptime=120)
    if len(res) == 0:
        raise ValueError(f"No 2-minute TESS light curve for TIC {tic}")
    sectors = [int(str(m).split()[-1]) for m in res.table["mission"]]
    if sector in (None, "", 0):
        i = 0
    elif int(sector) in sectors:
        i = sectors.index(int(sector))
    else:
        raise ValueError(f"Sector {sector} not available. 2-min sectors: {sorted(set(sectors))}")
    lc = res[i].download(quality_bitmask="default").remove_nans().normalize()
    return (np.asarray(lc.time.value, float), np.asarray(lc.flux.value, float),
            sectors[i], sorted(set(sectors)))


def group_detections(p_flare, p_tier, windows, starts, time, cfg, threshold):
    """Turn per-window model outputs into a table of flares.

    Overlapping windows see the same flare several times, so detections whose
    peaks are close together are merged into one flare:
      - flare_prob  = the highest probability among its windows
      - energy_tier = tier with the highest AVERAGE probability over its windows
    """
    det_cfg = cfg["detection"]
    length = windows.shape[1] if len(windows) else cfg["window"]["length"]
    ch1 = make_channels(windows)[:, :, 0] if len(windows) else np.empty((0, length))
    pos = ch1.argmax(axis=1)                       # where the peak is inside each window

    keep = ((p_flare >= threshold)
            & (pos >= int(det_cfg["peak_min_frac"] * length))
            & (pos <= int(det_cfg["peak_max_frac"] * length)))
    wi = np.where(keep)[0]
    if len(wi) == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    det = pd.DataFrame({"wi": wi, "p": p_flare[wi], "pos": pos[wi], "peak_idx": starts[wi] + pos[wi]})
    det = det.sort_values("peak_idx").reset_index(drop=True)
    det["event"] = (det["peak_idx"].diff() > det_cfg["merge_points"]).cumsum()

    rows = []
    for _, g in det.groupby("event"):
        best = g.loc[g["p"].idxmax()]
        tier_p = p_tier[g["wi"].values].mean(axis=0)
        snr, height = flare_snr(windows[int(best["wi"])], int(best["pos"]))
        rows.append(dict(
            peak_time_btjd=round(float(time[int(best["peak_idx"])]), 4),
            flare_prob=round(float(best["p"]), 3),
            energy_tier=TIER_NAMES[int(tier_p.argmax())],
            tier_conf=round(float(tier_p.max()), 2),
            amplitude=round(height, 4),
            snr=round(snr, 1) if np.isfinite(snr) else np.nan,
        ))
    ev = pd.DataFrame(rows)
    ev["weak"] = ev["snr"] < det_cfg["weak_snr"]
    return ev[EVENT_COLUMNS]


def scan_light_curve(model, time, flux, star_scaled, cfg, threshold):
    """Cut the light curve into windows, run the model once, and group the flares.

    Returns (events_table, number_of_windows).
    """
    w = cfg["window"]
    windows, starts = sliding_windows(time, flux, w["length"], w["stride"], w["cadence_days"])
    if len(windows) == 0:
        return pd.DataFrame(columns=EVENT_COLUMNS), 0
    out = model.predict(
        {"light_curve": make_channels(windows),
         "star_info": np.repeat(np.asarray(star_scaled, np.float32)[None], len(windows), axis=0)},
        batch_size=2048, verbose=0)
    ev = group_detections(out["flare"][:, 0], out["tier"], windows, starts, time, cfg, threshold)
    return ev, len(windows)


def catalogue_flares(tic, t0, t1, dedupe_days=0.005):
    """Known flare peak times (BTJD) from Günther (2020) and Lin (2024) between t0 and t1.

    The same flare is often in both catalogues, so peaks closer than dedupe_days are
    counted once. Needs internet; returns an empty array if the lookup fails.
    """
    from astroquery.vizier import Vizier

    Vizier.ROW_LIMIT = -1
    peaks = []
    try:
        g = Vizier(columns=["TIC", "tpeak"]).query_constraints(catalog="J/AJ/159/60/table1", TIC=str(tic))
        if len(g):
            peaks += list(np.asarray(g[0]["tpeak"], float) - 2457000)    # BJD -> BTJD
    except Exception:     # catalogue service down: carry on without it
        pass
    try:
        lin = Vizier(columns=["TIC", "tPeak"]).query_constraints(catalog="J/AJ/168/234/table9", TIC=str(tic))
        if len(lin):
            peaks += list(np.asarray(lin[0]["tPeak"], float))
    except Exception:
        pass
    return dedupe_peaks(np.array(peaks), t0, t1, dedupe_days)


def dedupe_peaks(peaks, t0, t1, dedupe_days=0.005):
    """Keep peaks inside [t0, t1] and merge ones closer than dedupe_days."""
    peaks = np.sort(np.asarray(peaks, float))
    peaks = peaks[(peaks >= t0) & (peaks <= t1)]
    if len(peaks) > 1:
        peaks = peaks[np.r_[True, np.diff(peaks) > dedupe_days]]
    return peaks


def match_catalogue(events, cat_peaks, tol_days):
    """Mark which detections are known catalogue flares, and count recovered catalogue flares."""
    events = events.copy()
    det_t = events["peak_time_btjd"].to_numpy(float)
    if len(cat_peaks) and len(det_t):
        events["in_catalogue"] = [bool(np.abs(cat_peaks - x).min() < tol_days) for x in det_t]
        recovered = int(sum(np.abs(det_t - c).min() < tol_days for c in cat_peaks))
    else:
        events["in_catalogue"] = False
        recovered = 0
    return events, recovered


def analyze_star(model, tic, cfg, mean, std, threshold, sector=None, teff=None):
    """Full pipeline for one star: star info -> light curve -> flares -> catalogue check."""
    from .star import star_info

    t_start = _time.time()
    tic = int(str(tic).upper().replace("TIC", "").strip())
    star = star_info(tic, mean, std, teff)
    t, f, sec, all_secs = download_light_curve(tic, sector)
    ev, n_win = scan_light_curve(model, t, f, star["scaled"], cfg, threshold)
    cat = catalogue_flares(tic, t[0], t[-1], cfg["catalogue"]["dedupe_days"])
    ev, recovered = match_catalogue(ev, cat, cfg["catalogue"]["match_tolerance_days"])
    return dict(tic=tic, sector=sec, all_sectors=all_secs, star=star, events=ev,
                time=t, flux=f, catalogue=cat, recovered=recovered,
                n_windows=n_win, seconds=round(_time.time() - t_start, 1))
