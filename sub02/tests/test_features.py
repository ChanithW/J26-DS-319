import numpy as np
import pandas as pd
import pytest

from component2.features import prep, shape_feats
from component2.windowing import cut_window, stellar_window, to_2min


def synth(tau=12.0, step=2.0, amp=5.0, bg=10.0):
    tm = np.arange(-20, 90, step)
    f = np.full_like(tm, bg)
    rise = (tm >= -4) & (tm < 0)
    f[rise] = bg + amp * (tm[rise] + 4) / 4
    dec = tm >= 0
    f[dec] = bg + amp * np.exp(-tm[dec] / tau)
    return tm, f


def test_recovers_tau():
    tm, f = synth(tau=12.0)
    r = shape_feats(tm, f)
    assert r is not None
    assert r["decay_tau_min"] == pytest.approx(12.0, rel=0.15)


def test_flat_and_empty_return_none():
    tm = np.arange(-20, 90, 2.0)
    assert shape_feats(tm, np.full_like(tm, 5.0)) is None
    assert shape_feats(np.array([]), np.array([])) is None


def test_prep_logs_time_features_only():
    df = pd.DataFrame({"rise10_min": [3.0], "g5": [0.5]})
    out = prep(df, ["rise10_min", "g5"])
    assert out["rise10_min"].iloc[0] == pytest.approx(np.log1p(3.0))
    assert out["g5"].iloc[0] == 0.5


def test_cut_window():
    t = np.arange(-30, 100.0)
    tt, _ = cut_window(t, t)
    assert tt.min() == -20 and tt.max() == 90


def test_to_2min_binning_and_gap_drop():
    t = np.arange(0, 10, 2 / 60)          # 60 points per 2 minutes
    f = np.ones_like(t)
    tb, fb = to_2min(t, f, phase=0.0)
    assert len(tb) == 5 and np.allclose(fb, 1.0)
    keep = (t < 4) | (t > 6)              # create a gap
    tb2, _ = to_2min(t[keep], f[keep], 0.0)
    assert len(tb2) == 4


def test_stellar_window_units():
    time = 100.0 + np.arange(-30, 100, 2) / 1440.0
    tm, f = stellar_window(time, np.ones_like(time), 100.0)
    assert tm.min() >= -20 and tm.max() <= 90
