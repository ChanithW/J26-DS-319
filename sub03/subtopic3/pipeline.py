"""
Glue between Stage 1 output and the Sub-topic 3 predictor.

    stage1 dict --(gate: is_flare must be True)--> resolve stellar parameters
        (dataset row for a known KIC, otherwise Stage-1 / manual parameters)
        --> FlarePredictor.predict_one --> stage3 dict (contract)
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Mapping

import pandas as pd

from . import config
from .contracts import STELLAR_PARAM_FIELDS, ContractError, validate_stage1_output
from .predictor import FlarePredictor


class GatingError(RuntimeError):
    """Stage 1 did not confirm a flare, so Sub-topic 3 must not run."""


def check_gate(stage1: Mapping[str, Any] | None) -> None:
    """Allow Sub-topic 3 only if Stage 1 exists, is valid, and is_flare is True."""
    if not stage1:
        raise GatingError("No Stage 1 result found. Run flare detection (Stage 1) first.")
    try:
        validate_stage1_output(dict(stage1))
    except ContractError as exc:
        raise GatingError(f"Stage 1 result is malformed: {exc}") from exc
    if stage1["is_flare"] is not True:
        raise GatingError(
            "Stage 1 classified this event as NOT a flare. Flare frequency/energy "
            "prediction is only run for confirmed flares."
        )


@lru_cache(maxsize=1)
def load_catalog(path: str | None = None) -> pd.DataFrame:
    """Read the stellar flare database once, indexed by KIC."""
    df = pd.read_csv(path or config.DATA_PATH)
    return df.set_index("KIC", drop=False)


def lookup_star(kic: int | None, catalog: pd.DataFrame | None = None) -> dict | None:
    """Stellar parameters for a KIC from the dataset, or None if unknown."""
    if kic is None:
        return None
    catalog = load_catalog() if catalog is None else catalog
    if kic not in catalog.index:
        return None
    row = catalog.loc[kic]
    params = {}
    for name in (*STELLAR_PARAM_FIELDS, "spectral_type"):
        value = row[name]
        params[name] = None if (isinstance(value, float) and math.isnan(value)) else value
    return params


def resolve_params(stage1: Mapping[str, Any], catalog: pd.DataFrame | None = None) -> tuple[dict, str]:
    """
    Choose the parameters to predict with.
    Returns (params, source) where source is 'dataset' or 'stage1'.
    A KIC found in the dataset wins; otherwise Stage 1's stellar_params are used.
    """
    from_dataset = lookup_star(stage1.get("KIC"), catalog)
    if from_dataset is not None:
        return from_dataset, "dataset"
    return dict(stage1.get("stellar_params") or {}), "stage1"


def run_stage3(
    stage1: Mapping[str, Any],
    predictor: FlarePredictor,
    params: Mapping[str, Any] | None = None,
) -> tuple[dict, list[str]]:
    """
    Full Sub-topic 3 step: gate, then predict.
    ``params`` lets the UI pass user-edited parameters; otherwise they are resolved
    from the dataset / Stage 1. Returns (stage3_output, warnings).
    """
    check_gate(stage1)
    if params is None:
        params, _ = resolve_params(stage1)
    return predictor.predict_one(params, kic=stage1.get("KIC"))
