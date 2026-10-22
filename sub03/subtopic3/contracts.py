"""
Pipeline interface contract between the 4 stages of the J26-DS-319 system.

    Stage 1 (Sub-topic 1)  flare / not-flare classification
        |  STAGE1_OUTPUT  ->  st.session_state["pipeline"]["stage1"]
        v
    Stage 2 (Sub-topic 3)  flare frequency + energy with 80 % intervals   <- this module
        |  STAGE3_OUTPUT  ->  st.session_state["pipeline"]["stage3"]
        v
    Stage 3 (Sub-topic 2)  CME prediction
        v
    Stage 4 (Sub-topic 4)  habitability risk (reads stage1 + stage3 + stage2)

Other members should code against the field names in this file (or the JSON
version written to ``sub03/contracts/stage_contract.json``). Field names are
STABLE: add new optional fields rather than renaming existing ones.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Shared-state keys (Streamlit st.session_state)
# ---------------------------------------------------------------------------
PIPELINE_KEY = "pipeline"   # st.session_state["pipeline"] is a dict holding:
STAGE1_KEY = "stage1"       #   ["stage1"] -> dict following STAGE1_OUTPUT_FIELDS
STAGE3_KEY = "stage3"       #   ["stage3"] -> dict following STAGE3_OUTPUT_FIELDS

SPECTRAL_TYPES = ("F", "G", "K", "M")

# Stellar parameters the Sub-topic 3 model uses (in the model's feature order,
# excluding spectral_code which is derived from spectral_type / Teff).
STELLAR_PARAM_FIELDS = (
    "Teff", "logg", "Prot", "radius_gspphot", "mh_gspphot",
    "bp_rp", "parallax", "phot_g_mean_mag",
)

# ---------------------------------------------------------------------------
# INPUT: what Sub-topic 1 must hand to Sub-topic 3
# Each entry: field -> (allowed python types, required?, description)
# ---------------------------------------------------------------------------
STAGE1_OUTPUT_FIELDS: dict[str, tuple[tuple[type, ...], bool, str]] = {
    "is_flare": ((bool,), True, "Stage-1 verdict. Sub-topic 3 runs only when True."),
    "KIC": ((int, type(None)), False, "Kepler Input Catalog ID of the star, if known."),
    "stellar_params": ((dict, type(None)), False,
                       "Optional dict of stellar parameters (keys from STELLAR_PARAM_FIELDS, "
                       "plus optional 'spectral_type'). Used when the KIC is not in the dataset."),
    "flare_probability": ((float, int, type(None)), False, "Classifier confidence for 'flare' (0-1)."),
    "flare_details": ((dict, type(None)), False, "Free-form details of the detected event."),
}

# ---------------------------------------------------------------------------
# OUTPUT: what Sub-topic 3 hands to Sub-topic 2 and Sub-topic 4
# ---------------------------------------------------------------------------
STAGE3_OUTPUT_FIELDS: dict[str, tuple[tuple[type, ...], bool, str]] = {
    "KIC": ((int, type(None)), True, "Star ID (None if parameters were entered manually)."),
    "spectral_type": ((str,), True, "One of F, G, K, M."),
    "pred_flare_rate": ((float,), True, "Predicted flare rate [flares/day] (point estimate)."),
    "pred_flare_rate_lower": ((float,), True, "Lower bound of the 80 % interval [flares/day]."),
    "pred_flare_rate_upper": ((float,), True, "Upper bound of the 80 % interval [flares/day]."),
    "pred_logE": ((float,), True, "Predicted median flare energy, log10(E / erg) (point estimate)."),
    "pred_logE_lower": ((float,), True, "Lower bound of the 80 % interval, log10(erg)."),
    "pred_logE_upper": ((float,), True, "Upper bound of the 80 % interval, log10(erg)."),
    "detection_bias_flag": ((bool,), True,
                            "True for F-type stars: Kepler selection bias inflates their "
                            "apparent rates/energies (observational, not physical)."),
    "interval_level": ((float,), False, "Nominal coverage of the intervals (0.80)."),
}


class ContractError(ValueError):
    """Raised when a payload does not follow the stage contract."""


def _validate(payload: Any, spec: dict, name: str) -> None:
    """Check ``payload`` against a field spec (presence of required fields + types).

    Args:
        payload: the dict to check.
        spec: one of STAGE1_OUTPUT_FIELDS / STAGE3_OUTPUT_FIELDS.
        name: label used in error messages ("stage1" / "stage3").
    Raises:
        ContractError: describing the first problem found.
    """
    if not isinstance(payload, dict):
        raise ContractError(f"{name} must be a dict, got {type(payload).__name__}.")
    for field, (types, required, _) in spec.items():
        # Missing optional fields are fine; missing required ones are not.
        if field not in payload:
            if required:
                raise ContractError(f"{name} is missing required field '{field}'.")
            continue
        value = payload[field]
        # bool is a subclass of int: do not accept True/False as a KIC or a number
        if isinstance(value, bool) and bool not in types:
            raise ContractError(f"{name}.{field} must not be a bool.")
        if not isinstance(value, types):
            allowed = ", ".join(t.__name__ for t in types)
            raise ContractError(f"{name}.{field} must be of type ({allowed}), got {type(value).__name__}.")


def validate_stage1_output(payload: Any) -> None:
    """Check a Stage-1 payload. Raises ContractError if it is malformed."""
    _validate(payload, STAGE1_OUTPUT_FIELDS, "stage1")


def validate_stage3_output(payload: Any) -> None:
    """Check a Sub-topic 3 payload, including interval ordering and finiteness."""
    _validate(payload, STAGE3_OUTPUT_FIELDS, "stage3")
    # Value-level checks on top of the type checks.
    if payload["spectral_type"] not in SPECTRAL_TYPES:
        raise ContractError(f"stage3.spectral_type must be one of {SPECTRAL_TYPES}.")
    # Both quantities: finite numbers in the order lower <= point <= upper.
    for base in ("pred_flare_rate", "pred_logE"):
        lo, mid, hi = payload[f"{base}_lower"], payload[base], payload[f"{base}_upper"]
        if not all(math.isfinite(v) for v in (lo, mid, hi)):
            raise ContractError(f"stage3.{base} values must be finite.")
        if not lo <= mid <= hi:
            raise ContractError(f"stage3.{base}: expected lower <= point <= upper, got {lo}, {mid}, {hi}.")
    # A rate comes from 10**x, so it can never be zero or negative.
    if payload["pred_flare_rate_lower"] <= 0:
        raise ContractError("stage3.pred_flare_rate_lower must be > 0.")


def contract_as_json() -> dict:
    """Language-neutral description of both payloads (written to stage_contract.json)."""
    def describe(spec):
        # Python types -> their names, so the result is plain JSON.
        return {
            field: {"types": [t.__name__ for t in types], "required": required, "description": desc}
            for field, (types, required, desc) in spec.items()
        }
    return {
        "session_state": {"root_key": PIPELINE_KEY, "stage1_key": STAGE1_KEY, "stage3_key": STAGE3_KEY},
        "stellar_param_fields": list(STELLAR_PARAM_FIELDS),
        "stage1_output": describe(STAGE1_OUTPUT_FIELDS),
        "stage3_output": describe(STAGE3_OUTPUT_FIELDS),
    }


if __name__ == "__main__":
    # python -m subtopic3.contracts  -> regenerates contracts/stage_contract.json
    out = Path(__file__).resolve().parents[1] / "contracts" / "stage_contract.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract_as_json(), indent=2))
    print(f"Wrote {out}")
