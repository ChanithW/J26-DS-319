"""Hand-off from Stage 1 (this component) to Stage 2 (sub03, flare frequency & energy).

The format follows sub03's stage contract (sub03/subtopic3/contracts.py and
sub03/contracts/stage_contract.json), so sub03 can read our result without changes:

    st.session_state["pipeline"]["stage1"] = {
        "is_flare":          bool          (required) - at least one flare found
        "KIC":               int | None    - Kepler ID; TESS stars have none -> None
        "stellar_params":    dict | None   - Teff, logg, bp_rp, parallax, phot_g_mean_mag
                                             from the TESS Input Catalog (others None)
        "flare_probability": float | None  - highest flare probability among the flares
        "flare_details":     dict | None   - our full result: TIC, sector, flare counts
                                             per energy tier, strongest tier, flare list
    }

The energy tiers are inside "flare_details", because sub03's contract has no field
for them yet. This file only builds and checks the payload; it does not change sub03.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .energy import TIER_NAMES

# Shared Streamlit session_state keys (same names as sub03)
PIPELINE_KEY = "pipeline"
STAGE1_KEY = "stage1"
STAGE3_KEY = "stage3"          # sub03's output; cleared when a new Stage-1 result is saved

# sub03's stellar parameters, in its order
STELLAR_PARAM_FIELDS = ("Teff", "logg", "Prot", "radius_gspphot", "mh_gspphot",
                        "bp_rp", "parallax", "phot_g_mean_mag")

# field -> (allowed types, required)
STAGE1_FIELDS = {
    "is_flare": ((bool,), True),
    "KIC": ((int, type(None)), False),
    "stellar_params": ((dict, type(None)), False),
    "flare_probability": ((float, int, type(None)), False),
    "flare_details": ((dict, type(None)), False),
}


class ContractError(ValueError):
    """The payload does not follow the Stage-1 contract."""


def _num_or_none(value):
    """Python float, or None for missing / NaN / inf (JSON-safe)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def stellar_params(star: dict) -> dict:
    """sub03's stellar parameters from our star info.

    Only values with the same meaning are filled. Prot, radius_gspphot and
    mh_gspphot are not in the TIC in the same form, so they stay None
    (sub03's model handles missing values).
    """
    params = {name: None for name in STELLAR_PARAM_FIELDS}
    params["Teff"] = _num_or_none(star.get("teff"))
    params["logg"] = _num_or_none(star.get("logg"))
    params["bp_rp"] = _num_or_none(star.get("bp_rp"))
    params["parallax"] = _num_or_none(star.get("parallax"))
    params["phot_g_mean_mag"] = _num_or_none(star.get("gaia_gmag"))
    if star.get("type") in ("F", "G", "K", "M"):
        params["spectral_type"] = star["type"]
    return params


def stage1_payload(result: dict) -> dict:
    """Build the Stage-1 payload from one analysed star (output of detect.analyze_star)."""
    ev, star = result["events"], result["star"]
    n = int(len(ev))
    counts = {name: int((ev["energy_tier"] == name).sum()) for name in TIER_NAMES} if n else \
        {name: 0 for name in TIER_NAMES}
    strongest = max(ev["energy_tier"], key=TIER_NAMES.index) if n else None
    flares = [
        {"peak_time_btjd": _num_or_none(r.peak_time_btjd),
         "flare_prob": _num_or_none(r.flare_prob),
         "energy_tier": str(r.energy_tier),
         "tier_conf": _num_or_none(r.tier_conf),
         "snr": _num_or_none(r.snr),
         "weak": bool(r.weak)}
        for r in ev.itertuples(index=False)
    ]
    payload = {
        "is_flare": bool(n > 0),
        "KIC": None,
        "stellar_params": stellar_params(star),
        "flare_probability": _num_or_none(ev["flare_prob"].max()) if n else None,
        "flare_details": {
            "source": "Component 1 - two-head 1D-CNN (TESS)",
            "tic_id": f"TIC {result['tic']}",
            "gaia_dr3_id": f"Gaia DR3 {star['gaia']}" if star.get("gaia") else None,
            "sector": int(result["sector"]),
            "star_type": star.get("type"),
            "teff_source": star.get("teff_source"),
            "n_flares": n,
            "n_weak": int(np.sum(ev["weak"])) if n else 0,
            "tier_counts": counts,
            "strongest_tier": strongest,
            "flares": flares,
        },
    }
    validate_stage1_payload(payload)
    return payload


def validate_stage1_payload(payload: Any) -> None:
    """Same type rules as sub03's validate_stage1_output. Raises ContractError."""
    if not isinstance(payload, dict):
        raise ContractError(f"stage1 must be a dict, got {type(payload).__name__}.")
    for field, (types, required) in STAGE1_FIELDS.items():
        if field not in payload:
            if required:
                raise ContractError(f"stage1 is missing required field '{field}'.")
            continue
        value = payload[field]
        if isinstance(value, bool) and bool not in types:      # bool is a subclass of int
            raise ContractError(f"stage1.{field} must not be a bool.")
        if not isinstance(value, types):
            allowed = ", ".join(t.__name__ for t in types)
            raise ContractError(f"stage1.{field} must be ({allowed}), got {type(value).__name__}.")
