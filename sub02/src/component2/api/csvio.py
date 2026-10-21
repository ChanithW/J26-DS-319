"""Long-format CSV parsing for POST /predict/file."""
from __future__ import annotations

import io

import pandas as pd

REQUIRED = ["flare_id", "star_id", "tpeak", "time", "flux"]


class CsvError(ValueError):
    """The uploaded file is not valid; the message is shown to the user."""


def parse_long_csv(raw: bytes) -> list[dict]:
    """One row per light-curve point; grouped by flare_id into request-format records."""
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise CsvError("The file could not be read as a CSV.") from e
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise CsvError(f"Missing column(s): {', '.join(missing)}. Expected: {', '.join(REQUIRED)}.")
    for c in ["star_id", "tpeak", "time", "flux"]:
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.isna().any():
            bad = int(conv.isna().to_numpy().argmax()) + 2
            raise CsvError(f"Column '{c}' must be numeric (first problem on line {bad}).")
        df[c] = conv
    if df["flare_id"].isna().any():
        raise CsvError("Column 'flare_id' has empty values.")
    records = []
    for fid, g in df.groupby("flare_id", sort=False):
        if len(g) < 2:
            raise CsvError(f"Flare '{fid}' has fewer than 2 rows.")
        records.append({"flare_id": str(fid), "star_id": int(g["star_id"].iloc[0]),
                        "tpeak": float(g["tpeak"].iloc[0]),
                        "time": g["time"].tolist(), "flux": g["flux"].tolist()})
    if not records:
        raise CsvError("The file has no data rows.")
    return records
