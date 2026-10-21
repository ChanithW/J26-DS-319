"""Unit tests for feature engineering (no internet, no model needed)."""
import numpy as np

from flare_cnn.features import flare_snr, make_channels, robust_std, sliding_windows, window_noise


def quiet_window(n=128, noise=0.001, seed=0):
    return 1 + np.random.default_rng(seed).normal(0, noise, n)


def add_flare(w, peak=50, amp=0.05, decay=5):
    """Fast rise, exponential decay - the classic flare shape."""
    w = w.copy()
    idx = np.arange(len(w)) - peak
    w[idx >= 0] += amp * np.exp(-idx[idx >= 0] / decay)
    return w


def test_robust_std_ignores_outliers():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 10_000)
    a[:50] = 1_000                          # a few huge outliers, like a flare
    assert abs(robust_std(a) - 1) < 0.05
    assert np.std(a) > 10                   # the normal std is ruined by them


def test_window_noise_recovers_true_noise():
    w = quiet_window(n=5000, noise=0.002)
    assert abs(window_noise(w)[0] - 0.002) / 0.002 < 0.1


def test_make_channels_shape_and_type():
    X = make_channels(np.stack([quiet_window(seed=s) for s in range(5)]))
    assert X.shape == (5, 128, 2)
    assert X.dtype == np.float32


def test_channel1_is_clipped():
    w = add_flare(quiet_window(noise=0.0005), amp=1.0)   # huge flare
    ch1 = make_channels(w)[0, :, 0]
    assert ch1.max() <= 60 and ch1.min() >= -10


def test_channel2_keeps_size_information():
    small = make_channels(add_flare(quiet_window(), amp=0.01))[0, :, 1].max()
    big = make_channels(add_flare(quiet_window(), amp=0.5))[0, :, 1].max()
    assert big > small                      # the energy head can tell them apart


def test_flat_window_does_not_divide_by_zero():
    X = make_channels(np.ones((1, 128)))
    assert np.isfinite(X).all()


def test_snr_weak_and_strong_flares():
    w = quiet_window(noise=0.001)
    snr_weak, _ = flare_snr(add_flare(w, amp=0.003), 50)
    snr_strong, _ = flare_snr(add_flare(w, amp=0.05), 50)
    assert snr_weak < 5 < snr_strong


def test_sliding_windows_skip_gaps():
    t = np.arange(1000) * 2 / 1440
    t[500:] += 1.0                          # a 1-day data gap (like the TESS orbit gap)
    f = np.ones(1000)
    W, starts = sliding_windows(t, f, 128, 16)
    for s in starts:                        # no window may cross the gap
        assert not (s < 500 < s + 128)
    assert np.allclose(np.median(W, axis=1), 1)
