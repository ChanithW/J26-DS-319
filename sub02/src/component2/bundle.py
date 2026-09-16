"""Versioned model bundle: save and load."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib

from . import BUNDLE_VERSION

log = logging.getLogger(__name__)

REQUIRED_KEYS = ["version", "model", "bootstrap_models", "feats", "pooled_median", "sd0", "alignment",
                 "platt", "base", "range_lo", "range_hi", "conformal", "config", "data_hash",
                 "library_versions", "created", "summary"]


class BundleError(RuntimeError):
    """Raised when a bundle is missing, malformed or the wrong version."""


def save_bundle(bundle: dict, path: str | Path) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in bundle]
    if missing:
        raise BundleError(f"bundle is missing keys: {missing}")
    joblib.dump(bundle, path)
    log.info("bundle saved to %s", path)


def load_bundle(path: str | Path) -> dict:
    """Load a bundle and check its version."""
    p = Path(path)
    if not p.exists():
        raise BundleError(f"bundle not found: {p}")
    b = joblib.load(p)
    if not isinstance(b, dict) or b.get("version") != BUNDLE_VERSION:
        got = b.get("version") if isinstance(b, dict) else type(b).__name__
        raise BundleError(f"bundle version mismatch: expected {BUNDLE_VERSION!r}, got {got!r}. "
                          "Older notebook bundles (isotonic calibration) are not supported; retrain.")
    missing = [k for k in REQUIRED_KEYS if k not in b]
    if missing:
        raise BundleError(f"bundle is missing keys: {missing}")
    return b
