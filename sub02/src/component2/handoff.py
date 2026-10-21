"""Hand-off table for Sub-topic 4.

The column list is NOT confirmed with Sub-topic 4; keep all changes in this module.
Spectral type is deliberately not produced by this component.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HANDOFF_COLUMNS = ["flare_id", "star_id", "calibrated_probability", "confidence_lower",
                   "confidence_upper", "predicted_label", "reliability"]


def write_handoff(pred: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Write the hand-off CSV (status == ok rows only) and return it."""
    out = pred.loc[pred["status"] == "ok", HANDOFF_COLUMNS]
    out.to_csv(path, index=False)
    return out
