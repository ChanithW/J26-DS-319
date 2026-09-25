"""Input and output schemas (kept in one place so they are easy to change)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FlareInput(BaseModel):
    flare_id: str
    star_id: int
    tpeak: float = Field(description="Flare peak time, BTJD (Julian date - 2457000)")
    time: Optional[list[float]] = Field(default=None, description="Days, BTJD (omit only when fetching is enabled)")
    flux: Optional[list[float]] = Field(default=None, description="Any positive scale")


class PredictRequest(BaseModel):
    flares: list[FlareInput]
    save_run: bool = False
    run_name: Optional[str] = None


class Curve(BaseModel):
    t_min: list[float]
    g: list[float]


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
    features: Optional[dict[str, Optional[float]]] = None
    curve: Optional[Curve] = None


class RunSummary(BaseModel):
    n_flares: int
    n_ok: int
    n_eruptive: int
    n_confined: int
    n_confident: int
    n_uncertain: int


class RunResponse(BaseModel):
    run_id: Optional[str] = None
    name: Optional[str] = None
    model_version: str
    created_at: str
    summary: RunSummary
    results: list[FlareResult]


class RunListItem(BaseModel):
    run_id: str
    name: Optional[str] = None
    created_at: str
    model_version: str
    n_flares: int
    n_eruptive: int
    n_confident: int


class HealthResponse(BaseModel):
    status: str
    model_version: str


class ModelInfo(BaseModel):
    bundle_version: str
    created: str
    data_hash: str
    training_summary: dict
    metrics: dict
    limitations: list[str]
