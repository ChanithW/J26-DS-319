"""Loaders and validation for the solar and stellar data files."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

SOLAR_RAW_COLS = ["timestamp", "xrsa", "xrsb", "xrsa_quality", "xrsb_quality"]
LABEL_COLS = ["flare_id", "flare_key", "peak_time", "start_time", "end_time", "goes_class",
              "peak_flux", "eruptive", "lat", "lon", "ar_number", "phi_ar", "ar_area"]
TESS_RAW_COLS = ["tmin", "flux"]


class DataError(ValueError):
    """Raised when a data file is missing or malformed."""


def _require_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(f"{where}: missing columns {missing}")


def load_solar_labels(data_dir: str | Path) -> pd.DataFrame:
    """Load and validate the solar label table (the source of truth for flares)."""
    path = Path(data_dir) / "solar_labels" / "labels_seed_standardized.csv"
    if not path.exists():
        raise DataError(f"solar label file not found: {path}")
    df = pd.read_csv(path)
    _require_cols(df, LABEL_COLS, str(path))
    if df["flare_id"].duplicated().any():
        raise DataError(f"{path}: duplicate flare_id values")
    if df[["eruptive", "ar_number", "peak_time"]].isna().any().any():
        raise DataError(f"{path}: missing eruptive/ar_number/peak_time values")
    if not set(df["eruptive"].unique()) <= {0, 1}:
        raise DataError(f"{path}: 'eruptive' must be 0/1")
    df["peak_time"] = pd.to_datetime(df["peak_time"])
    df["eruptive"] = df["eruptive"].astype(int)
    return df


def solar_raw_path(data_dir: str | Path, flare_id) -> Path:
    return Path(data_dir) / "solar_raw" / f"flare_{flare_id}.csv"


def warn_unlabelled_solar_files(data_dir: str | Path, labels: pd.DataFrame) -> list[str]:
    """Log a warning naming raw solar files that have no label row; they are ignored."""
    folder = Path(data_dir) / "solar_raw"
    if not folder.exists():
        raise DataError(f"solar raw folder not found: {folder}")
    known = {f"flare_{i}.csv" for i in labels["flare_id"]}
    extra = sorted(p.name for p in folder.glob("flare_*.csv") if p.name not in known)
    for name in extra:
        log.warning("solar raw file without a label (ignored): %s", name)
    return extra


def load_solar_raw(data_dir: str | Path, flare_id) -> pd.DataFrame:
    """Load one raw solar light curve."""
    path = solar_raw_path(data_dir, flare_id)
    if not path.exists():
        raise DataError(f"solar raw file not found: {path}")
    df = pd.read_csv(path)
    _require_cols(df, SOLAR_RAW_COLS, str(path))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_tess_features(data_dir: str | Path) -> pd.DataFrame:
    """Load the 13-feature table of the 33 stellar sample flares."""
    from .features import FEATS13

    path = Path(data_dir) / "tess_sample_features2.csv"
    if not path.exists():
        raise DataError(f"stellar feature file not found: {path}")
    df = pd.read_csv(path)
    _require_cols(df, FEATS13 + ["TIC"], str(path))
    return df


def load_tess_raw(data_dir: str | Path, tic) -> pd.DataFrame:
    """Load one raw TESS sample curve (tmin in minutes from the peak, flux)."""
    path = Path(data_dir) / "tess_raw" / f"tic_{tic}.csv"
    if not path.exists():
        raise DataError(f"TESS raw file not found: {path}")
    df = pd.read_csv(path)
    _require_cols(df, TESS_RAW_COLS, str(path))
    return df
