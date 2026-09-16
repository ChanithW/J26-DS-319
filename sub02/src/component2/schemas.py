"""Input and output schemas (kept in one place so they are easy to change)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FlareInput(BaseModel):
    flare_id: str
    star_id: int
    tpeak: float = Field(description="Flare peak time, BTJD (Julian date - 2457000)")
    time: list[float] = Field(description="Days, BTJD")
    flux: list[float] = Field(description="Any positive scale")


class PredictRequest(BaseModel):
    flares: list[FlareInput]


class FlareResult(BaseModel):
    flare_id: Optional[str] = None
    star_id: Optional[int] = None
    status: Literal["ok", "too_few_points", "no_peak", "invalid_input"]
    calibrated_probability: Optional[float] = None
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    predicted_label: Optional[Literal["eruptive", "confined"]] = None
    reliability: Optional[Literal["confident", "uncertain"]] = None
    features_missing: Optional[int] = None
    out_of_range: Optional[bool] = None
    warnings: list[str] = []
    raw_score: Optional[float] = None
    conformal_set: Optional[Literal["E", "C", "E+C", "none"]] = None
