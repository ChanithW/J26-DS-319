"""
Central configuration for Sub-topic 3 (flare frequency & energy prediction).

All paths are resolved relative to the ``sub03/`` folder so the code runs
from any working directory. Paths can be overridden with environment
variables (useful when the integrated app lives somewhere else).
"""
from __future__ import annotations

import os
from pathlib import Path

# sub03/ folder (this file lives in sub03/subtopic3/config.py)
SUB03_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(os.environ.get("SUB03_MODEL_PATH", SUB03_ROOT / "flare_model.pkl"))
DATA_PATH = Path(
    os.environ.get("SUB03_DATA_PATH", SUB03_ROOT / "data" / "stellar_flare_database_v3.csv")
)
OUTPUT_DIR = Path(os.environ.get("SUB03_OUTPUT_DIR", SUB03_ROOT / "outputs"))

# --------------------------------------------------------------------------
# Spectral classification (Pecaut & Mamajek 2013), as used in training.
# Lower Teff bound (inclusive) for each type, checked from hottest to coolest.
# --------------------------------------------------------------------------
SPECTRAL_TEFF_LOWER = {"F": 6000.0, "G": 5300.0, "K": 3900.0, "M": float("-inf")}
SPECTRAL_CODE = {"F": 0, "G": 1, "K": 2, "M": 3}  # matches pickle's 'spectral_code'

# Valid Teff domain. Training dropped A-type stars (Teff >= 7500 K, see the
# 01_attrition table); 2600 K is the coolest star in the training data.
TEFF_MIN = 2600.0
TEFF_MAX_EXCLUSIVE = 7500.0

# Training sample was dwarfs only (logg >= 4). Values below this are allowed
# but produce an out-of-domain warning.
LOGG_DWARF_MIN = 4.0

# Kepler long-cadence baseline used to compute flares_per_day = Nf / baseline.
BASELINE_DAYS = 1459.0

# Original split (reproduced and verified in scripts/evaluate.py):
# 20 % test set, stratified by spectral_type, random_state = 42.
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
