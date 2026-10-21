"""Tests for the model design and the flare-grouping step."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")   # skip if TensorFlow is not installed

from flare_cnn.detect import dedupe_peaks, group_detections, match_catalogue  # noqa: E402
from flare_cnn.model import build_model  # noqa: E402

CFG = {"window": {"length": 128, "stride": 16, "cadence_days": 2 / 1440},
       "detection": {"peak_min_frac": 0.2, "peak_max_frac": 0.6, "merge_points": 10, "weak_snr": 5}}


def test_model_inputs_and_outputs():
    m = build_model()
    out = m.predict({"light_curve": np.zeros((3, 128, 2), np.float32),
                     "star_info": np.zeros((3, 3), np.float32)}, verbose=0)
    assert out["flare"].shape == (3, 1)
    assert out["tier"].shape == (3, 4)
    assert np.allclose(out["tier"].sum(axis=1), 1, atol=1e-5)          # softmax


def test_split_model_has_separate_top_blocks():
    shared = build_model(tier_weight=1.0, split_last=False)
    split = build_model(tier_weight=2.0, split_last=True)
    n_conv = lambda m: sum(isinstance(l, tf.keras.layers.Conv1D) for l in m.layers)
    assert n_conv(shared) == 4 and n_conv(split) == 5                  # one extra block 4


def test_overlapping_windows_become_one_flare():
    t = np.arange(400) * 2 / 1440
    flux = np.ones(400)
    flux[200:210] += 0.05 * np.exp(-np.arange(10) / 3)
    starts = np.array([136, 152, 168])               # three windows that all contain the flare
    windows = np.stack([flux[s:s + 128] for s in starts])
    p_flare = np.array([0.9, 0.95, 0.8])
    p_tier = np.array([[0.1, 0.7, 0.1, 0.1]] * 3)
    ev = group_detections(p_flare, p_tier, windows, starts, t, CFG, threshold=0.6)
    assert len(ev) == 1
    assert ev.loc[0, "flare_prob"] == 0.95 and ev.loc[0, "energy_tier"] == "M"


def test_catalogue_dedupe_and_match():
    peaks = dedupe_peaks([10.000, 10.002, 12.0, 50.0], t0=0, t1=20)
    assert list(peaks) == [10.0, 12.0]                # duplicate merged, out-of-range removed
    import pandas as pd
    ev = pd.DataFrame({"peak_time_btjd": [10.001, 15.0]})
    ev, rec = match_catalogue(ev, peaks, tol_days=10 / 1440)
    assert list(ev["in_catalogue"]) == [True, False] and rec == 1
