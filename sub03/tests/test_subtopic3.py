"""
Tests for Sub-topic 3. Run from sub03/:   python -m pytest tests -q
"""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subtopic3 import config  # noqa: E402
from subtopic3.contracts import (  # noqa: E402
    STAGE3_OUTPUT_FIELDS, ContractError, validate_stage1_output, validate_stage3_output,
)
from subtopic3.pipeline import GatingError, check_gate, resolve_params, run_stage3  # noqa: E402
from subtopic3.predictor import FlarePredictor, InputError, prepare_input, spectral_type_from_teff  # noqa: E402

SUN_LIKE = {"Teff": 5770, "logg": 4.44, "Prot": 25.0}


@pytest.fixture(scope="module")
def predictor():
    return FlarePredictor()


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(config.DATA_PATH)


# ------------------------------------------------------------- contracts
def test_stage1_schema_accepts_valid_payload():
    validate_stage1_output({"is_flare": True, "KIC": 123, "stellar_params": None})


@pytest.mark.parametrize("payload", [
    {},                                   # missing is_flare
    {"is_flare": "yes"},                  # wrong type
    {"is_flare": True, "KIC": "abc"},     # KIC must be int
    {"is_flare": True, "KIC": True},      # bool is not a KIC
])
def test_stage1_schema_rejects_bad_payload(payload):
    with pytest.raises(ContractError):
        validate_stage1_output(payload)


def test_stage3_schema_rejects_crossed_interval(predictor):
    out, _ = predictor.predict_one(SUN_LIKE)
    bad = dict(out, pred_logE_lower=out["pred_logE"] + 1)
    with pytest.raises(ContractError):
        validate_stage3_output(bad)


def test_output_contract_fields(predictor):
    out, _ = predictor.predict_one(SUN_LIKE, kic=42)
    required = {k for k, (_, req, _) in STAGE3_OUTPUT_FIELDS.items() if req}
    assert required <= set(out)
    assert set(out) <= set(STAGE3_OUTPUT_FIELDS)
    assert out["KIC"] == 42 and out["spectral_type"] == "G"
    validate_stage3_output(out)


# ------------------------------------------------------- input validation
@pytest.mark.parametrize("teff", [7500, 9000, 30000, 2000])
def test_teff_out_of_range_rejected(teff):
    with pytest.raises(InputError):
        spectral_type_from_teff(teff)


@pytest.mark.parametrize("teff,expected", [
    (7499, "F"), (6000, "F"), (5999, "G"), (5300, "G"), (5299, "K"), (3900, "K"), (3899, "M"), (2600, "M"),
])
def test_spectral_type_boundaries(teff, expected):
    assert spectral_type_from_teff(teff) == expected


def test_inconsistent_spectral_type_rejected():
    with pytest.raises(InputError):
        prepare_input({**SUN_LIKE, "spectral_type": "M"})


def test_missing_logg_rejected():
    with pytest.raises(InputError):
        prepare_input({"Teff": 5000})


def test_optional_features_may_be_missing(predictor):
    out, warnings = predictor.predict_one({"Teff": 3500, "logg": 4.8})
    assert out["spectral_type"] == "M"
    assert any("Missing" in w for w in warnings)


# ------------------------------------------------------------ predictions
def test_quantile_ordering_on_dataset(predictor, dataset):
    res = predictor.predict_frame(dataset)
    assert res["error"].isna().all()
    for base in ("pred_flare_rate", "pred_logE"):
        lo, mid, hi = (res[f"{base}{s}"].astype(float) for s in ("_lower", "", "_upper"))
        assert (lo <= mid).all() and (mid <= hi).all()
    assert (res["pred_flare_rate_lower"].astype(float) > 0).all()


def test_detection_bias_flag_only_for_f_type(predictor):
    f_out, _ = predictor.predict_one({"Teff": 6500, "logg": 4.3})
    k_out, _ = predictor.predict_one({"Teff": 4500, "logg": 4.6})
    assert f_out["detection_bias_flag"] is True
    assert k_out["detection_bias_flag"] is False


def test_smoke_10_dataset_rows(predictor, dataset):
    rows = dataset.sample(10, random_state=0)
    res = predictor.predict_frame(rows)
    assert len(res) == 10 and res["error"].isna().all()
    assert (res["KIC"].astype(int).to_numpy() == rows["KIC"].to_numpy()).all()
    assert (res["spectral_type"].to_numpy() == rows["spectral_type"].to_numpy()).all()
    for rec in res.drop(columns="error").to_dict("records"):
        rec["detection_bias_flag"] = bool(rec["detection_bias_flag"])
        for k in rec:
            if k.startswith("pred_") or k == "interval_level":
                rec[k] = float(rec[k])
        validate_stage3_output(rec)
        assert all(math.isfinite(rec[k]) for k in rec if k.startswith("pred_"))


# ----------------------------------------------------------------- gating
@pytest.mark.parametrize("stage1", [None, {}, {"is_flare": False, "KIC": 1}, {"KIC": 1}])
def test_gating_blocks_when_not_flare(stage1, predictor):
    with pytest.raises(GatingError):
        check_gate(stage1)
    with pytest.raises(GatingError):
        run_stage3(stage1, predictor, params=SUN_LIKE)


def test_known_kic_autofills_from_dataset(dataset, predictor):
    kic = int(dataset["KIC"].iloc[0])
    params, source = resolve_params({"is_flare": True, "KIC": kic})
    assert source == "dataset" and params["Teff"] == dataset["Teff"].iloc[0]
    out, _ = run_stage3({"is_flare": True, "KIC": kic}, predictor)
    assert out["KIC"] == kic


def test_unknown_kic_uses_stage1_params():
    params, source = resolve_params({"is_flare": True, "KIC": 1, "stellar_params": SUN_LIKE})
    assert source == "stage1" and params == SUN_LIKE
