"""Load project settings from configs/config.yaml and the files saved with the model."""

import json
from pathlib import Path

import numpy as np
import yaml

# Repo root = two folders above this file (src/flare_cnn/config.py -> repo/)
ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    """Read config.yaml and return it as a dictionary.

    Relative paths inside the file are turned into full paths from the repo root,
    so the code works no matter which folder you run it from.
    """
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: str(ROOT / v) for k, v in cfg["paths"].items()}
    return cfg


def load_star_scaling(cfg):
    """Mean and std used to scale [Teff, log10 radius, log10 distance] during training.

    They come from prep_settings.json (saved by notebook 02). Using the same numbers
    at prediction time is essential: different scaling = wrong energy tiers.
    """
    with open(cfg["paths"]["prep_settings"], encoding="utf-8") as fh:
        prep = json.load(fh)
    return np.array(prep["scaling"]["mean"], float), np.array(prep["scaling"]["std"], float)


def load_threshold(cfg):
    """Detection threshold saved with the model (falls back to the value in config.yaml)."""
    try:
        with open(cfg["paths"]["model_settings"], encoding="utf-8") as fh:
            return float(json.load(fh)["threshold"])
    except (OSError, KeyError, ValueError):
        return float(cfg["detection"]["threshold"])
