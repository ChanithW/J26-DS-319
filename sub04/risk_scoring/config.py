"""Static configuration for the Sub-topic 4 habitability risk scoring app.

Constants only: no logic, no I/O, and the training CSV is never read at import time.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "xgb_dose_model.joblib"

FEATURES = ["koi_sma", "flare_rate_month", "flare_rate_all", "max_logE", "star_type"]
CLASSES = ["Low", "Medium", "High"]

# Formula label settings (dose proxy, Tilley-style)
D_REF = 0.16  # AU, reference orbital distance
THRESH_HIGH = 1.0
THRESH_MEDIUM = 0.1
DEFAULT_CME_PROB = 0.47

# Star type from Teff: above 6000 K = 0 (F), above 5200 K = 1 (G), above 3700 K = 2 (K), else 3 (M)
TEFF_CUTS = (6000, 5200, 3700)
STAR_TYPE_LETTERS = {0: "F", 1: "G", 2: "K", 3: "M"}

# TRAIN_RANGES: min/max over the TRAIN rows (split == "train", 950 rows) of
# D:\J26-DS-319\sub04\data\processed\training_v12_dose.csv
# Computed once with pandas 3.0.3 and hard-coded here, so the app never reads the CSV.
TRAIN_RANGES = {
    "koi_sma": (0.0059, 9.9),
    "flare_rate_month": (0.0, 11.450303139606332),
    "flare_rate_all": (0.0196078431372549, 48.03628476486142),
    "max_logE": (30.716837723299523, 37.600972895686745),
    "star_type": (0.0, 3.0),
    "Teff": (2790.0, 9196.7),
    "cme_prob": (0.0745364081101825, 0.4324161639899621),
}

LIMITS_TEXT = "This is a physics-based risk score, not measured ozone loss."
