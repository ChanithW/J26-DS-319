from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from component2 import BUNDLE_VERSION
from component2.bundle import BundleError, load_bundle, save_bundle
from component2.conformal import nonconformity, q_plain, q_weighted
from component2.features import FEATS13
from component2.predict import OUTPUT_COLUMNS, predict_features, predict_records
from component2.train import train_from_tables
from tests.conftest import make_table


def test_split_never_separates_active_region(tables):
    solar, _ = tables
    y, g = solar["eruptive"].values, solar["ar_number"].values
    folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=0).split(solar, y, g))
    parts = [set(g[f[1]]) for f in folds]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not parts[i] & parts[j]


def test_train_never_uses_stellar_as_training(tables, small_cfg):
    solar, stellar = tables
    b, rep = train_from_tables(solar, stellar, small_cfg)
    assert b["summary"]["n_train"] + b["summary"]["n_cal"] + b["summary"]["n_test"] == len(solar)
    assert rep["split"]["train"] == b["summary"]["n_train"]


def test_conformal_coverage_close_to_target():
    rs = np.random.RandomState(0)
    n = 4000
    y = (rs.rand(n) < 0.35).astype(int)
    p = np.clip(0.35 + 0.25 * (y - 0.35) + rs.normal(0, 0.15, n), 0.01, 0.99)
    cal, te = slice(0, 2000), slice(2000, n)
    nc = nonconformity(p[cal], y[cal])
    q1 = q_plain(nc[y[cal] == 1], 0.1)
    q0 = q_plain(nc[y[cal] == 0], 0.1)
    cov1 = ((1 - p[te]) <= q1)[y[te] == 1].mean()
    cov0 = (p[te] <= q0)[y[te] == 0].mean()
    assert abs(cov1 - 0.9) < 0.04 and abs(cov0 - 0.9) < 0.04
    # weighted with unit weights must agree closely with the plain threshold
    w = np.ones(int((y[cal] == 1).sum()))
    assert q_weighted(nc[y[cal] == 1], w, 1.0, 0.1) == pytest.approx(q1, abs=0.01)


def test_q_plain_inf_when_too_few():
    assert q_plain([0.1, 0.2], 0.1) == float("inf")


def test_output_schema_and_rows(bundle):
    stellar = make_table(5, 5, labelled=False)
    out = predict_features(bundle, stellar)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert (out["status"] == "ok").all()
    assert out["calibrated_probability"].between(0, 1).all()
    assert (out["confidence_lower"] <= out["confidence_upper"]).all()
    assert set(out["conformal_set"]) <= {"E", "C", "E+C", "none"}
    assert set(out["reliability"]) <= {"confident", "uncertain"}


def test_missing_features_and_out_of_range_are_uncertain(bundle):
    base = make_table(3, 7, labelled=False)
    many = base.copy()
    many.loc[0, FEATS13[:8]] = np.nan                    # 8 missing > 5
    far = base.copy()
    far.loc[1, "eff_dur_min"] = 1e9
    out = predict_features(bundle, pd.concat([many.iloc[[0]], far.iloc[[1]]]))
    assert out.loc[0, "features_missing"] == 8 and "many_features_missing" in out.loc[0, "warnings"]
    assert out.loc[0, "reliability"] == "uncertain"
    assert bool(out.loc[1, "out_of_range"]) and "out_of_range" in out.loc[1, "warnings"]
    assert out.loc[1, "reliability"] == "uncertain"


def curve(tpeak=100.0, tau=15.0, n=56):
    tm = np.arange(-20, 90, 2.0)[:n]
    f = np.full(len(tm), 10.0)
    r = (tm >= -4) & (tm < 0)
    f[r] = 10 + 5 * (tm[r] + 4) / 4
    d = tm >= 0
    f[d] = 10 + 5 * np.exp(-tm[d] / tau)
    return {"flare_id": "X", "star_id": 1, "tpeak": tpeak, "time": (tpeak + tm / 1440).tolist(), "flux": f.tolist()}


def test_predict_records_statuses(bundle):
    good = curve()
    few = {**curve(), "flare_id": "few", "time": curve()["time"][:5], "flux": curve()["flux"][:5]}
    flat = {**curve(), "flare_id": "flat", "flux": [5.0] * 55}
    bad = {"flare_id": "bad", "star_id": 3}
    out = predict_records(bundle, [good, few, flat, bad])
    assert list(out["status"]) == ["ok", "too_few_points", "no_peak", "invalid_input"]
    assert out.loc[1:, "calibrated_probability"].isna().all()
    assert list(out.columns) == OUTPUT_COLUMNS


def test_bundle_roundtrip_and_version_check(bundle, tmp_path):
    p = tmp_path / "b.joblib"
    save_bundle(bundle, p)
    assert load_bundle(p)["version"] == BUNDLE_VERSION
    bad = {**bundle, "version": "0"}
    save_bundle(bad, p)
    with pytest.raises(BundleError):
        load_bundle(p)


def test_training_is_repeatable(tables, small_cfg):
    solar, stellar = tables
    b1, _ = train_from_tables(solar, stellar, small_cfg)
    b2, _ = train_from_tables(solar, stellar, small_cfg)
    s = make_table(4, 9, labelled=False)
    a, b = predict_features(b1, s), predict_features(b2, s)
    assert np.allclose(a["calibrated_probability"], b["calibrated_probability"], atol=1e-3)


GOLDEN = Path(__file__).parent / "golden" / "handoff_v3_expected.csv"
DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(not (GOLDEN.exists() and (DATA / "tess_sample_features2.csv").exists()),
                    reason="golden file and real data not provided yet")
def test_golden():
    """Assumes flare_id == 'TESS_<TIC>' and star_id == TIC in the sample feature table."""
    from component2.config import load_config
    from component2.train import train

    bundle, _ = train(DATA, load_config())
    feats = pd.read_csv(DATA / "tess_sample_features2.csv")
    feats["flare_id"] = "TESS_" + feats["TIC"].astype(str)
    feats["star_id"] = feats["TIC"]
    out = predict_features(bundle, feats).merge(pd.read_csv(GOLDEN), on="flare_id", suffixes=("", "_exp"))
    assert len(out) == 33
    assert (out["calibrated_probability"] - out["calibrated_probability_exp"]).abs().max() < 0.002
    assert (out["confidence_lower"] - out["confidence_lower_exp"]).abs().max() < 0.01
    assert (out["confidence_upper"] - out["confidence_upper_exp"]).abs().max() < 0.01
    assert (out["predicted_label"] == out["predicted_label_exp"]).all()
    assert (out["reliability"] == out["reliability_exp"]).all()
