"""Load the model once and analyse stars. No Streamlit here, so it can be tested and reused.

    TIC ID -> star info (TIC) -> TESS light curve -> two-head CNN -> flare table
           -> catalogue check -> Stage-1 payload for sub03
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import load_config, load_star_scaling, load_threshold
from .contracts import stage1_payload
from .export import flares_table, star_summary


@dataclass
class Assets:
    """Everything the model needs, loaded once."""
    model: object
    cfg: dict
    mean: object
    std: object
    threshold: float


def missing_model_files(cfg: dict) -> list[str]:
    """Names of the required files that are not in models/."""
    p = cfg["paths"]
    files = {"best.keras": p["model"], "settings.json": p["model_settings"],
             "prep_settings.json": p["prep_settings"]}
    return [name for name, path in files.items() if not Path(path).exists()]


def load_assets(cfg: dict | None = None) -> Assets:
    """Read config, scaling and threshold, and load the trained model."""
    cfg = cfg or load_config()
    missing = missing_model_files(cfg)
    if missing:
        raise FileNotFoundError(f"Missing in models/: {', '.join(missing)}")
    from .model import load_model          # TensorFlow is imported only here
    mean, std = load_star_scaling(cfg)
    return Assets(load_model(cfg["paths"]["model"]), cfg, mean, std, load_threshold(cfg))


def analyse(assets: Assets, tic, sector=None, teff=None, analyzer=None) -> dict:
    """Analyse one star/sector. Returns the raw result plus tables and the payload.

    analyzer: same arguments as detect.analyze_star; tests pass a fake one so they
    run without internet.
    """
    if analyzer is None:
        from .detect import analyze_star as analyzer
    r = analyzer(assets.model, tic, assets.cfg, assets.mean, assets.std, assets.threshold,
                 sector=sector, teff=teff)
    return {
        "result": r,
        "flares": flares_table(r),
        "summary": star_summary(r),
        "payload": stage1_payload(r),
        "warnings": star_warnings(r["star"]),
    }


def star_warnings(star: dict) -> list[str]:
    """Plain-language warnings shown next to the result."""
    w = []
    if star.get("missing"):
        w.append(f"Missing star info ({', '.join(star['missing'])}): the training average was "
                 "used, so the energy tier is less reliable.")
    if "estimated" in str(star.get("teff_source")):
        w.append("Teff was not in the TESS Input Catalog, so it was estimated from the star's colour.")
    logg = star.get("logg")
    if star.get("type") == "unknown" or (logg is not None and pd.notna(logg) and logg < 4):
        w.append("This star is outside the training range (giant or unknown type): "
                 "results are experimental.")
    return w
