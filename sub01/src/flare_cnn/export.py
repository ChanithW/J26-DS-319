"""Save Component 1 results as CSV files for the other components.

Two outputs:
  1. per-flare CSV  - one row per detected flare
  2. star summary   - one row per star (flare counts per tier)

IDs are written as text ("TIC 123", "Gaia DR3 456...") so Excel does not turn
19-digit Gaia IDs into rounded numbers like 4.56E+18.
"""

import numpy as np
import pandas as pd

from .energy import TIER_NAMES

FLARE_COLUMNS = ["flare_id", "tic_id", "gaia_dr3_id", "sector", "peak_time_btjd",
                 "flare_prob", "energy_tier", "tier_conf", "amplitude_pct", "snr", "weak",
                 "in_catalogue"]


def flares_table(result):
    """Per-flare table for one analysed star/sector (output of detect.analyze_star)."""
    ev, star, tic, sec = result["events"], result["star"], result["tic"], result["sector"]
    df = ev.copy()
    df.insert(0, "flare_id", [f"C1_{tic}_S{sec}_{k + 1:03d}" for k in range(len(df))])
    df["tic_id"] = f"TIC {tic}"
    df["gaia_dr3_id"] = f"Gaia DR3 {star['gaia']}" if star["gaia"] else ""
    df["sector"] = sec
    df["amplitude_pct"] = (df["amplitude"] * 100).round(2)
    if "in_catalogue" not in df:
        df["in_catalogue"] = False
    return df[FLARE_COLUMNS]


def star_summary(result):
    """One summary row per star/sector (flare counts per tier, strongest tier)."""
    ev, star, t = result["events"], result["star"], result["time"]
    days = float(t[-1] - t[0]) if len(t) > 1 else np.nan
    counts = {f"n_{name}": int((ev["energy_tier"] == name).sum()) for name in TIER_NAMES}
    n = len(ev)
    return {
        "tic_id": f"TIC {result['tic']}",
        "gaia_dr3_id": f"Gaia DR3 {star['gaia']}" if star["gaia"] else "",
        "sector": result["sector"],
        "star_type": star["type"],
        "teff_K": round(star["teff"], 0) if np.isfinite(star["teff"]) else np.nan,
        "teff_source": star["teff_source"],
        "observed_days": round(days, 2),
        "n_flares": n,
        "n_weak": int(ev["weak"].sum()) if n else 0,
        **counts,
        "strongest_tier": max(ev["energy_tier"], key=TIER_NAMES.index) if n else "",
        "catalogue_flares": len(result["catalogue"]),
        "catalogue_recovered": result["recovered"],
        "missing_star_info": ", ".join(star["missing"]),
    }


def save_csv(df, path):
    """Save with UTF-8 BOM so Excel shows Günther and other accents correctly."""
    pd.DataFrame(df).to_csv(path, index=False, encoding="utf-8-sig")
    return path
