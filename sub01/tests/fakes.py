"""A fake analyser with the same output as flare_cnn.detect.analyze_star (no internet, no model)."""
import numpy as np
import pandas as pd


def fake_analyzer(model, tic, cfg, mean, std, threshold, sector=None, teff=None):
    tic = int(str(tic).upper().replace("TIC", "").strip())
    if tic == 999:
        raise ValueError("TIC 999 not found in the TESS Input Catalog")
    t = np.arange(0, 2, 2 / 1440)
    events = pd.DataFrame({
        "peak_time_btjd": [0.5, 1.2], "flare_prob": [0.97, 0.71],
        "energy_tier": ["M", "C"], "tier_conf": [0.8, 0.6],
        "amplitude": [0.02, 0.003], "snr": [25.0, 3.2], "weak": [False, True],
        "in_catalogue": [True, False],
    })
    star = dict(tic=tic, teff=3100.0 if teff is None else teff, teff_source="TIC", radius=0.3,
                distance=9.7, gaia="5853498713190525696", type="M", scaled=np.zeros(3, np.float32),
                missing=[], logg=5.0, bp_rp=2.8, parallax=np.float64(103.0), gaia_gmag=np.nan)
    return dict(tic=tic, sector=sector or 27, all_sectors=[13, 27], star=star, events=events,
                time=t, flux=np.ones_like(t), catalogue=np.array([0.5001]), recovered=1,
                n_windows=100, seconds=1.0)


class FakeAssets:
    model, cfg, mean, std, threshold = object(), {}, np.zeros(3), np.ones(3), 0.6
