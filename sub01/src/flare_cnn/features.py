"""Feature engineering: from raw flux windows to the 2 channels the CNN reads.

Same code as notebooks 02 and 06, moved here so training, the app and the tests
all use exactly one version.
"""

import numpy as np


def robust_std(a, axis=-1):
    """Standard deviation that ignores outliers (such as flares).

    Uses the median absolute deviation (MAD). 1.4826 makes it equal to the normal
    standard deviation for Gaussian noise.
    """
    a = np.asarray(a, float)
    med = np.median(a, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(a - med), axis=axis)


def window_noise(windows):
    """Noise level of each window from point-to-point jumps.

    Using differences (np.diff) ignores slow starspot trends, so we do not need to
    detrend the light curve. Dividing by sqrt(2) because a difference of two noisy
    points has sqrt(2) times the noise of one point.
    """
    windows = np.atleast_2d(windows)
    return robust_std(np.diff(windows, axis=1), axis=1) / np.sqrt(2)


def normalise_windows(windows):
    """Divide each window by its median so the quiet level is 1.0."""
    windows = np.atleast_2d(np.asarray(windows, float))
    return windows / np.median(windows, axis=1, keepdims=True)


def make_channels(windows):
    """Build the model input, shape (n_windows, 128, 2).

    Channel 1 = (flux - 1) / noise, clipped to [-10, 60]
        "how many noise levels above normal" -> helps with weak flares on any star.
    Channel 2 = arcsinh((flux - 1) / 0.01)
        the real brightness change, compressed so huge superflares do not dominate
        -> keeps the size information the energy head needs.
    """
    windows = np.atleast_2d(np.asarray(windows, float))
    noise = np.maximum(window_noise(windows), 1e-5)[:, None]   # avoid dividing by 0
    ch1 = np.clip((windows - 1) / noise, -10, 60)
    ch2 = np.arcsinh((windows - 1) / 0.01)
    return np.stack([ch1, ch2], axis=-1).astype(np.float32)


def flare_snr(window, peak):
    """Signal-to-noise ratio of a flare peaking at index `peak` in one window.

    height = highest point near the peak minus the local level just before the flare
    noise  = point-to-point noise away from the flare
    A flare with SNR < 5 is counted as weak.
    Returns (snr, height).
    """
    w = np.asarray(window, float)
    p = int(peak)
    away = np.r_[w[:max(p - 3, 0)], w[min(p + 15, len(w)):]]
    d = np.diff(away) if len(away) > 10 else np.diff(w)
    noise = robust_std(d) / np.sqrt(2)
    pre = w[max(p - 15, 0):max(p - 2, 1)]
    base = np.median(pre) if len(pre) >= 3 else np.median(w)
    height = w[max(p - 2, 0):p + 3].max() - base
    snr = height / noise if noise > 0 else np.nan
    return float(snr), float(height)


def sliding_windows(time, flux, length=128, stride=16, cadence=2 / 1440):
    """Cut a light curve into overlapping windows, skipping any that cross a data gap.

    Returns (windows, start_indices). Windows are already normalised to median 1.
    """
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    if len(time) <= length:
        return np.empty((0, length)), np.empty(0, int)
    starts = np.arange(0, len(time) - length, stride)
    no_gap = (time[starts + length - 1] - time[starts]) <= (length - 1) * cadence * 1.1
    starts = starts[no_gap]
    if len(starts) == 0:
        return np.empty((0, length)), starts
    windows = np.stack([flux[s:s + length] for s in starts])
    return normalise_windows(windows), starts
