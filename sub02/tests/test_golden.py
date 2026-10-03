"""Golden tests against the notebook's 33-flare hand-off table (tests/golden/handoff_v3_expected.csv).

1. ``test_golden_downstream_exact``: the notebook model's raw scores are frozen in
   ``frozen_scores_v3.json``. Everything after the XGBoost model (Platt, base-rate label, weighted
   conformal reliability) is run by the package and must reproduce the table exactly.
2. ``test_golden_retrained_agrees``: retrain on the real data. XGBoost output depends on the
   library version and thread count, so only a looser agreement is required (see MODEL_CARD).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from component2.calibration import Platt
from component2.conformal import WeightsModel, nonconformity
from component2.features import FEATS13, TIMEF
from component2.predict import predict_features

HERE = Path(__file__).parent / "golden"
GOLDEN = HERE / "handoff_v3_expected.csv"
FROZEN = HERE / "frozen_scores_v3.json"
DATA = Path(__file__).resolve().parents[1] / "data"


class FrozenModel:
    """Stands in for the XGBoost model: returns the frozen raw scores for the frozen features."""

    def __init__(self, L, p):
        self.L, self.p = np.asarray(L), np.asarray(p)

    def predict_proba(self, X):
        assert np.allclose(X, self.L, atol=1e-9), "features differ from the frozen ones"
        return np.c_[1 - self.p, self.p]


@pytest.mark.skipif(not (GOLDEN.exists() and FROZEN.exists()), reason="golden files not present")
def test_golden_downstream_exact():
    fz = json.loads(FROZEN.read_text())
    Lt = np.array(fz["stellar_scaled_features"])
    p_st, p_ca, y_ca = np.array(fz["stellar_raw_score"]), np.array(fz["cal_raw_score"]), np.array(fz["cal_label"])
    A_ca = np.array(fz["cal_aligned_features"])

    platt = Platt().fit(p_ca, y_ca)
    wm = WeightsModel().fit(A_ca, Lt)
    bundle = {
        "feats": FEATS13, "pooled_median": pd.Series(0.0, index=FEATS13), "sd0": pd.Series(1.0, index=FEATS13),
        "model": FrozenModel(Lt, p_st), "bootstrap_models": [FrozenModel(Lt, p_st)], "platt": platt,
        "base": float(y_ca.mean()), "range_lo": np.full(13, -np.inf), "range_hi": np.full(13, np.inf),
        "config": {"conformal_weighted": True, "label_rule": "platt_base", "max_missing_features": 5},
        "conformal": {"nc_ca": nonconformity(p_ca, y_ca), "y_ca": y_ca, "w_ca": wm(A_ca), "weights_model": wm,
                      "q1": None, "q0": None, "alpha": 0.10},
    }
    raw = Lt.copy()                                         # undo the log step so the package's prep recreates Lt
    for j, c in enumerate(FEATS13):
        if c in TIMEF:
            raw[:, j] = np.expm1(Lt[:, j])
    df = pd.DataFrame(raw, columns=FEATS13)
    df["flare_id"] = fz["flare_id"]
    df["star_id"] = [int(f.split("_")[1]) for f in fz["flare_id"]]

    out = predict_features(bundle, df).merge(pd.read_csv(GOLDEN), on="flare_id", suffixes=("", "_exp"))
    assert len(out) == 33
    assert (out["calibrated_probability"] - out["calibrated_probability_exp"]).abs().max() < 0.002
    assert (out["predicted_label"] == out["predicted_label_exp"]).all()
    assert (out["reliability"] == out["reliability_exp"]).all()


@pytest.mark.skipif(not (GOLDEN.exists() and (DATA / "tess_sample_features2.csv").exists()),
                    reason="golden file and real data not present")
def test_golden_retrained_agrees():
    """Assumes flare_id == 'TESS_<TIC>' and star_id == TIC in the sample feature table."""
    from component2.config import load_config
    from component2.train import train

    bundle, _ = train(DATA, load_config())
    feats = pd.read_csv(DATA / "tess_sample_features2.csv")
    feats["flare_id"] = "TESS_" + feats["TIC"].astype(str)
    feats["star_id"] = feats["TIC"]
    out = predict_features(bundle, feats).merge(pd.read_csv(GOLDEN), on="flare_id", suffixes=("", "_exp"))
    assert len(out) == 33
    assert spearmanr(out["calibrated_probability"], out["calibrated_probability_exp"])[0] > 0.85
    assert (out["calibrated_probability"] - out["calibrated_probability_exp"]).abs().max() < 0.12
    assert (out["predicted_label"] == out["predicted_label_exp"]).sum() >= 29
    assert "TESS_294750180" in set(out.nlargest(3, "calibrated_probability")["flare_id"])
    assert (out["confidence_lower"] - out["confidence_lower_exp"]).abs().max() < 0.08
    assert (out["confidence_upper"] - out["confidence_upper_exp"]).abs().max() < 0.08
