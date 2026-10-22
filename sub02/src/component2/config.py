"""Configuration loading."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

DEFAULT_YAML = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass
class Config:
    """All tunable settings; defaults mirror the research runs."""

    alpha: float = 0.10
    split_seed: int = 0
    split_folds: int = 4
    n_bootstrap: int = 50
    bootstrap_seed: int = 0
    phase_seed: int = 0
    window_min: tuple[float, float] = (-20.0, 90.0)
    min_points: int = 20
    min_bin_points: int = 20
    alignment: str = "diag"
    conformal_weighted: bool = True
    label_rule: str = "platt_base"
    max_missing_features: int = 5
    out_of_range_check: bool = False
    range_percentiles: tuple[float, float] = (0.5, 99.5)
    fetch_lightcurves: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path | None = None) -> Config:
    """Load a config from YAML (falls back to defaults if no file is found)."""
    p = Path(path) if path else DEFAULT_YAML
    cfg = Config()
    if p.exists():
        raw = yaml.safe_load(p.read_text()) or {}
        names = {f.name for f in fields(Config)}
        unknown = set(raw) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        for k, v in raw.items():
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    if cfg.alignment not in ("none", "diag", "coral"):
        raise ValueError("alignment must be none, diag or coral")
    if cfg.label_rule not in ("platt_base", "raw_0.5"):
        raise ValueError("label_rule must be platt_base or raw_0.5")
    return cfg
