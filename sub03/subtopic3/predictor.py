"""
Inference for Sub-topic 3: flare frequency and flare energy with 80 % intervals.

The pickle (``flare_model.pkl``) is a dict produced by the training notebook:

    targets   : ["log_flares_per_day" (XGBoost), "median_logE" (LightGBM)]
    models[t] : {"features": [...9 names...],
                 "point": [5 models, squared-error],
                 "lo":    [5 models, quantile 0.10],
                 "hi":    [5 models, quantile 0.90],
                 "qhat":  {"global", "F", "G", "K", "M"} conformal (CQR) corrections}
    alpha     : 0.2  -> 80 % prediction intervals

Prediction for one target (same as training):
    point = mean(point models)
    lower = mean(lo models) - qhat[spectral_type]
    upper = mean(hi models) + qhat[spectral_type]
then lower <= point <= upper is enforced. The flare rate is predicted in
log10(flares/day) and back-transformed with 10**x.

Preprocessing: none is bundled. The 9 raw features are passed as-is; missing
values stay NaN (XGBoost/LightGBM handle them natively, as in training).

Flow for one star:
    raw parameters --prepare_input()--> PreparedInput (validated, typed)
        --_design_matrix()--> feature table in the model's column order
        --_predict_target()--> point / lower / upper on the model (log) scale
        --_to_contract()--> output dict in natural units (flares/day, log10 erg)
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import config
from .contracts import STELLAR_PARAM_FIELDS, validate_stage3_output

# Keys of the two targets inside the pickle's "models" dict.
RATE_TARGET = "log_flares_per_day"   # log10(flares per day), XGBoost
ENERGY_TARGET = "median_logE"        # log10(median flare energy / erg), LightGBM


class InputError(ValueError):
    """Raised for invalid stellar parameters (shown to the user as-is)."""


# ===========================================================================
# 1. Input validation and preprocessing
# ===========================================================================
def spectral_type_from_teff(teff: float) -> str:
    """Pecaut & Mamajek (2013) class used in training. Rejects O/B/A and too-cool stars.

    Args:
        teff: effective temperature in kelvin.
    Returns:
        "F", "G", "K" or "M".
    Raises:
        InputError: if Teff is missing, A-type or hotter (>= 7500 K) or cooler
        than the coolest training star (< 2600 K).
    """
    # Teff is mandatory: the spectral type (a model feature) is derived from it.
    if teff is None or not math.isfinite(teff):
        raise InputError("Teff is required to determine the spectral type.")
    # Out-of-domain stars are rejected rather than extrapolated.
    if teff >= config.TEFF_MAX_EXCLUSIVE:
        raise InputError(
            f"Teff = {teff:.0f} K is A-type or hotter (>= {config.TEFF_MAX_EXCLUSIVE:.0f} K). "
            "The model only covers F, G, K and M dwarfs."
        )
    if teff < config.TEFF_MIN:
        raise InputError(
            f"Teff = {teff:.0f} K is below the coolest training star ({config.TEFF_MIN:.0f} K)."
        )
    # Thresholds are ordered hottest -> coolest, so the first match is the class.
    for sp_type, lower in config.SPECTRAL_TEFF_LOWER.items():  # hottest first
        if teff >= lower:
            return sp_type
    raise InputError(f"Could not classify Teff = {teff}.")  # unreachable


def _to_float(name: str, value: Any) -> float:
    """Convert a user/CSV value to float; None / '' / NaN become NaN (= missing)."""
    # Blank form fields and empty CSV cells mean "unknown", stored as NaN.
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        # Text such as "abc" in a numeric field: report which field is wrong.
        raise InputError(f"{name} must be a number, got {value!r}.") from None


@dataclass
class PreparedInput:
    """Validated, model-ready stellar parameters for one star."""
    features: dict[str, float]          # raw feature name -> value (NaN = missing)
    spectral_type: str                  # "F", "G", "K" or "M" (derived from Teff)
    warnings: list[str] = field(default_factory=list)  # non-fatal notes for the user


def prepare_input(params: Mapping[str, Any]) -> PreparedInput:
    """Validate one star's parameters and derive spectral_type from Teff if absent.

    Args:
        params: mapping with any of the STELLAR_PARAM_FIELDS (plus an optional
            'spectral_type'); missing keys are treated as unknown values.
    Returns:
        PreparedInput with numeric features, the spectral type and warnings.
    Raises:
        InputError: for any value the model cannot be used with.
    """
    # Read every model parameter; anything absent becomes NaN.
    values = {name: _to_float(name, params.get(name)) for name in STELLAR_PARAM_FIELDS}
    warnings: list[str] = []

    # Spectral type always comes from Teff (as in training).
    sp_type = spectral_type_from_teff(values["Teff"])
    # If the caller also supplied a type, it must be valid and agree with Teff.
    given = params.get("spectral_type")
    if given not in (None, "") and not (isinstance(given, float) and math.isnan(given)):
        given = str(given).strip().upper()
        if given not in config.SPECTRAL_CODE:
            raise InputError(f"spectral_type '{given}' is not supported; only F, G, K, M.")
        if given != sp_type:
            raise InputError(
                f"spectral_type '{given}' disagrees with Teff = {values['Teff']:.0f} K "
                f"(which gives '{sp_type}'). Training derived the type from Teff."
            )

    # logg was never missing in training, so it is required here as well.
    if math.isnan(values["logg"]):
        raise InputError("logg is required (it was never missing in the training data).")
    # Giants (logg < 4) were excluded from training: allowed, but flagged.
    if values["logg"] < config.LOGG_DWARF_MIN:
        warnings.append(
            f"logg = {values['logg']:.2f} < {config.LOGG_DWARF_MIN}: the model was trained on "
            "dwarfs only, so this prediction is an extrapolation."
        )
    # Physical sanity checks on optional values that were provided.
    if not math.isnan(values["Prot"]) and values["Prot"] <= 0:
        raise InputError("Prot (rotation period, days) must be positive.")
    if not math.isnan(values["radius_gspphot"]) and values["radius_gspphot"] <= 0:
        raise InputError("radius_gspphot must be positive.")

    # Tell the user which optional values were missing (they stay NaN).
    missing = [n for n in STELLAR_PARAM_FIELDS if math.isnan(values[n])]
    if missing:
        warnings.append(
            "Missing (handled natively by the model, as in training): " + ", ".join(missing)
        )
    return PreparedInput(features=values, spectral_type=sp_type, warnings=warnings)


# ---------------------------------------------------------------------------
# Loading the pickle
# ---------------------------------------------------------------------------
# Each pickled XGBoost booster is a UBJSON buffer {"Config": {...}, "Model": {...}}.
# With xgboost 3.4.1 on Windows, xgboost's own unpickling rejects these buffers
# ("input stream corrupted"), although the "Model" part is a complete model in
# XGBoost's official save_model format and loads cleanly. The fallback below
# loads that part IN MEMORY - the .pkl file itself is never modified.
# Correctness is verified by scripts/evaluate.py (test RMSE must equal training).
#
# UBJSON key marker for "Model": 'L' + 8-byte big-endian length (5) + the name.
_UBJ_MODEL_KEY = b"L\x00\x00\x00\x00\x00\x00\x00\x05Model"


def _make_compat_booster_class():
    """Build a Booster subclass whose unpickling falls back to the 'Model' section.

    xgboost is imported here (not at module top) so it is only needed when a
    pickle containing XGBoost models is actually loaded.
    """
    import xgboost as xgb
    from xgboost.core import XGBoostError

    class CompatBooster(xgb.Booster):
        def __setstate__(self, state: dict) -> None:
            # 1) Try xgboost's normal unpickling first (works on most setups).
            try:
                super().__setstate__(dict(state))
                return
            except XGBoostError as exc:
                original_error = exc
            # 2) Fallback: locate the "Model" section inside the raw buffer.
            raw = bytes(state["handle"])
            start = raw.find(_UBJ_MODEL_KEY)
            if start < 0 or not raw.endswith(b"}"):
                # Not the expected layout: report xgboost's own error.
                raise original_error
            # The model object runs from after the key to just before the
            # buffer's final '}' (which closes the outer {"Config", "Model"} object).
            model_bytes = bytearray(raw[start + len(_UBJ_MODEL_KEY):-1])  # drop outer '}'
            # Load it with XGBoost's official model loader, in memory.
            xgb.Booster.__init__(self, model_file=model_bytes)
            # Restore the remaining Python-side attributes of the booster.
            self.__dict__.update({k: v for k, v in state.items() if k != "handle"})

    return CompatBooster


class _ModelUnpickler(pickle.Unpickler):
    """Standard unpickler, except XGBoost boosters use the fallback above."""

    def find_class(self, module: str, name: str):
        # Swap in the compatible Booster class; everything else loads normally.
        if module == "xgboost.core" and name == "Booster":
            return _make_compat_booster_class()
        return super().find_class(module, name)


# ===========================================================================
# 2. Predictor
# ===========================================================================
class FlarePredictor:
    """Loads the pickle once and predicts flare rate / energy with intervals."""

    def __init__(self, model_path: str | Path | None = None):
        """Load the model bundle (default path from config.MODEL_PATH)."""
        path = Path(model_path or config.MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as fh:
            # Own trained artefact; needs xgboost==3.4.1 and lightgbm==4.6.0.
            self.bundle: dict = _ModelUnpickler(fh).load()
        self.models: dict = self.bundle["models"]
        # alpha = 0.2 in the pickle -> 80 % prediction intervals.
        self.interval_level = 1.0 - float(self.bundle["alpha"])
        # Both targets use the same 9 features in the same order.
        self.feature_order: list[str] = list(self.models[RATE_TARGET]["features"])

    # ---------------------------------------------------------------- core
    def _design_matrix(self, prepared: list[PreparedInput]) -> pd.DataFrame:
        """Build the feature table (one row per star) in the model's column order."""
        rows = []
        for p in prepared:
            row = dict(p.features)
            # Encode the spectral type as in training: F=0, G=1, K=2, M=3.
            row["spectral_code"] = config.SPECTRAL_CODE[p.spectral_type]
            rows.append(row)
        # Column order must match training exactly; NaN stays NaN.
        X = pd.DataFrame(rows)[self.feature_order].astype(float)
        X["spectral_code"] = X["spectral_code"].astype(int)  # trained as int
        return X

    def _predict_target(self, target: str, X: pd.DataFrame, sp_types: list[str]):
        """Point estimate and 80 % interval for one target, on the model (log) scale.

        Returns:
            (point, lower, upper) as NumPy arrays, one value per row of X.
        """
        m = self.models[target]
        # Ensemble: average the 5 models of each kind.
        point = np.mean([mod.predict(X) for mod in m["point"]], axis=0)
        lo = np.mean([mod.predict(X) for mod in m["lo"]], axis=0)    # 10 % quantile
        hi = np.mean([mod.predict(X) for mod in m["hi"]], axis=0)    # 90 % quantile
        # Conformalised quantile regression (CQR): widen the raw quantile band by
        # the calibrated correction for this spectral type (global if unknown).
        qhat = np.array([m["qhat"].get(t, m["qhat"]["global"]) for t in sp_types])
        lower, upper = lo - qhat, hi + qhat
        # Enforce lower <= point <= upper (quantile models can cross the mean model).
        lower = np.minimum(lower, point)
        upper = np.maximum(upper, point)
        return point, lower, upper

    def predict_log_frame(self, prepared: list[PreparedInput]) -> pd.DataFrame:
        """Predictions on the model scale (log10 rate, logE) - used by evaluation."""
        X = self._design_matrix(prepared)
        types = [p.spectral_type for p in prepared]
        r_pt, r_lo, r_hi = self._predict_target(RATE_TARGET, X, types)    # flare rate
        e_pt, e_lo, e_hi = self._predict_target(ENERGY_TARGET, X, types)  # flare energy
        return pd.DataFrame({
            "spectral_type": types,
            "log_rate": r_pt, "log_rate_lower": r_lo, "log_rate_upper": r_hi,
            "logE": e_pt, "logE_lower": e_lo, "logE_upper": e_hi,
        })

    # ------------------------------------------------------------- public
    def predict_one(self, params: Mapping[str, Any], kic: int | None = None) -> tuple[dict, list[str]]:
        """Predict for one star. Returns (stage3_output_dict, warnings).

        Args:
            params: stellar parameters (see prepare_input).
            kic: optional Kepler ID, copied into the output.
        Raises:
            InputError: if the parameters are invalid.
        """
        prepared = prepare_input(params)
        row = self.predict_log_frame([prepared]).iloc[0]
        output = self._to_contract(row, kic)
        # Guarantee the output follows the documented contract before returning it.
        validate_stage3_output(output)
        return output, prepared.warnings

    def predict_frame(self, df: pd.DataFrame, kic_column: str = "KIC") -> pd.DataFrame:
        """Batch prediction. Invalid rows get an 'error' message instead of predictions.

        Returns:
            DataFrame aligned with ``df`` holding the output-contract columns plus
            an 'error' column (empty for rows that were predicted).
        """
        # Validate every row first; collect valid ones and error messages separately,
        # so one bad row does not stop the whole batch.
        prepared, ok_idx, kics, errors = [], [], {}, {}
        for idx, rec in zip(df.index, df.to_dict("records")):
            try:
                kics[idx] = _clean_kic(rec.get(kic_column))
                prepared.append(prepare_input(rec))
                ok_idx.append(idx)
            except InputError as exc:
                errors[idx] = str(exc)

        out_cols = list(_OUTPUT_COLUMNS)
        result = pd.DataFrame(index=df.index, columns=out_cols + ["error"], dtype=object)
        # Predict all valid rows in one vectorised call, then fill their rows.
        if prepared:
            log_df = self.predict_log_frame(prepared)
            log_df.index = ok_idx
            for idx, row in log_df.iterrows():
                result.loc[idx, out_cols] = pd.Series(self._to_contract(row, kics[idx]))
        # Rejected rows keep their KIC (if any) and the reason.
        for idx, msg in errors.items():
            result.at[idx, "KIC"] = kics.get(idx)
            result.at[idx, "error"] = msg
        return result.infer_objects()  # numeric columns back to float/bool dtypes

    def _to_contract(self, row: pd.Series, kic: int | None) -> dict:
        """Convert one model-scale row into the output contract (natural units)."""
        return {
            "KIC": _clean_kic(kic),
            "spectral_type": str(row["spectral_type"]),
            # Back-transform log10(flares/day) -> flares/day (monotonic, so the
            # interval order is preserved).
            "pred_flare_rate": float(10 ** row["log_rate"]),
            "pred_flare_rate_lower": float(10 ** row["log_rate_lower"]),
            "pred_flare_rate_upper": float(10 ** row["log_rate_upper"]),
            # Energy stays on the log10(erg) scale, as in the dataset.
            "pred_logE": float(row["logE"]),
            "pred_logE_lower": float(row["logE_lower"]),
            "pred_logE_upper": float(row["logE_upper"]),
            # Kepler selection bias toward Sun-like stars (see README).
            "detection_bias_flag": row["spectral_type"] == "F",
            "interval_level": round(self.interval_level, 4),
        }


# Column order of batch outputs (same fields as the Sub-topic 3 output contract).
_OUTPUT_COLUMNS = (
    "KIC", "spectral_type",
    "pred_flare_rate", "pred_flare_rate_lower", "pred_flare_rate_upper",
    "pred_logE", "pred_logE_lower", "pred_logE_upper",
    "detection_bias_flag", "interval_level",
)


def _clean_kic(value: Any) -> int | None:
    """KIC as a plain Python int (or None) so it serialises and passes the contract."""
    # Blank / NaN KIC = star entered without an ID.
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
        return None
    try:
        # Also converts NumPy integers and floats such as 8226464.0 from CSVs.
        return int(value)
    except (TypeError, ValueError):
        raise InputError(f"KIC must be an integer, got {value!r}.") from None
