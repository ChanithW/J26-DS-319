"""Component 1 pipeline: flare detection + energy prediction for one light curve.

Inputs : a light curve (time, flux, flux_err) + Teff (K), radius (R_sun), contamination ratio
Outputs: one row per flare (probability, brightening, SNR, energy, type, confidence level)
The steps are exactly the ones used in training and in C1_03_predict.
"""
import json
import re
import numpy as np
import pandas as pd

W, STEP, GAP = 128, 16, 10 / 1440          # window (points), step (points), max gap inside a window (days)
JOIN_MIN = 20                              # peaks closer than this (minutes) = same detection
TYPES = ['Solar-like', 'Large', 'Superflare', 'Extreme superflare']
CUTS = [32, 33, 34]                        # log10 erg boundaries (Shibata 2013; Günther 2020; Tilley 2019)
COLOURS = {'Solar-like': '#2e9e5b', 'Large': '#d39e00',
           'Superflare': '#e8743b', 'Extreme superflare': '#c62828'}


# ---------------------------------------------------------------- loading
def load_assets(model_dir='model'):
    from tensorflow import keras
    model = keras.models.load_model(f'{model_dir}/best_v2.keras')
    prep = json.load(open(f'{model_dir}/prep_settings.json'))
    levels = json.load(open(f'{model_dir}/app_settings.json'))
    return model, prep, levels


def parse_rule(rule):
    """'2 pts ≥ 2.5σ' -> (2, 2.5); 'no rule' -> (0, 0)."""
    nums = re.findall(r'[\d.]+', rule)
    return (int(nums[0]), float(nums[1])) if len(nums) >= 2 else (0, 0.0)


# ---------------------------------------------------------------- inputs
def read_light_curve(file):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {'time', 'flux', 'flux_err'} - set(df.columns)
    if missing:
        raise ValueError(f'CSV is missing column(s): {", ".join(sorted(missing))}')
    df = df[['time', 'flux', 'flux_err']].apply(pd.to_numeric, errors='coerce').dropna()
    df = df[(df.flux > 0) & (df.flux_err > 0)].sort_values('time').reset_index(drop=True)
    if len(df) < W:
        raise ValueError(f'Light curve too short: {len(df)} points (need at least {W}).')
    return df


def check_inputs(teff, radius, contam, cadence_min):
    warn = []
    if contam > 0.1:
        warn.append('Contamination > 0.1: light from nearby stars may affect the results.')
    if not 2500 <= teff <= 7500:
        warn.append('Teff is outside the F/G/K/M dwarf range the model was trained on.')
    if radius > 2:
        warn.append('Radius > 2 R☉: this may be a giant star; the model was trained on dwarfs.')
    if abs(cadence_min - 2) > 0.5:
        warn.append(f'Cadence is {cadence_min:.1f} min; the model was trained on 2-minute TESS data.')
    return warn


def star_type(teff):
    return 'M' if teff < 3900 else 'K' if teff < 5300 else 'G' if teff < 6000 else 'F'


# ---------------------------------------------------------------- core steps
def _run_len(z, j, sig):
    """Number of consecutive points around j that are above sig."""
    if z[j] <= sig:
        return 0
    a = j
    while a > 0 and z[a - 1] > sig:
        a -= 1
    b = j
    while b < len(z) - 1 and z[b + 1] > sig:
        b += 1
    return b - a + 1


def detect(t, fx, fe, teff, radius, model, prep, sigmas, pmin=0.3):
    """Windows -> model -> candidate flares (one row per joined detection)."""
    starts = np.array([s for s in range(0, len(t) - W + 1, STEP) if np.diff(t[s:s + W]).max() <= GAP])
    n_windows = len(starts)
    if n_windows == 0:
        return pd.DataFrame(), 0
    idx = starts[:, None] + np.arange(W)
    med = np.median(fx[idx], axis=1, keepdims=True)          # each window: normal brightness = 1
    exc, er = fx[idx] / med - 1, fe[idx] / med
    z = exc / np.maximum(er, 1e-6)
    X = np.stack([np.clip(z, -10, 60), np.arcsinh(exc / 0.01)], -1).astype(np.float32)
    s1 = (np.array([teff, np.log10(radius)]) - np.array(prep['star_mean'])) / np.array(prep['star_std'])
    S = np.tile(s1.astype(np.float32), (n_windows, 1))
    out = model.predict({'light_curve': X, 'star': S}, batch_size=2048, verbose=0)
    pw = out['flare'][:, 0]
    logE = out['energy'][:, 0] * prep['logE_std'] + prep['logE_mean']

    pk = [(idx[w, j], z[w, j], pw[w]) for w in np.where(pw >= pmin)[0] for j in [int(np.argmax(z[w]))]]
    if not pk:
        return pd.DataFrame(), n_windows
    pk = pd.DataFrame(pk, columns=['i', 'sig', 'p']).sort_values('i')
    groups, cur = [], []
    for r in pk.itertuples():
        if cur and t[r.i] - t[cur[-1].i] > JOIN_MIN / 1440:
            groups.append(cur)
            cur = []
        cur.append(r)
    groups.append(cur)

    rows = []
    for g in groups:
        i0 = max(g, key=lambda r: r.sig).i
        pos = i0 - starts
        cand = np.where((pos >= 0) & (pos < W))[0]
        w = cand[np.argmin(np.abs(pos[cand] - 0.4 * W))]       # peak at ~40% of the window, as in training
        j = pos[w]
        row = dict(peak_time=t[i0], flare_prob=max(r.p for r in g),
                   peak_brightening_pct=100 * exc[w, j], snr=z[w, j], log10_energy=float(logE[w]))
        for s in sigmas:
            row[f'run_{s}'] = _run_len(z[w], j, s)
        rows.append(row)
    return pd.DataFrame(rows), n_windows


def apply_levels(c, levels):
    """Label each candidate 'confirmed' / 'candidate' (settings tuned on validation light curves)."""
    def passes(name):
        lv = levels[name]
        n, s = parse_rule(lv['rule'])
        ok = c.flare_prob >= lv['threshold']
        return ok & (c[f'run_{s}'] >= n) if n else ok
    conf = passes('confirmed')
    cand = passes('candidate') & ~conf
    c = c[conf | cand].copy()
    c['level'] = np.where(conf[conf | cand], 'confirmed', 'candidate')
    return c


def merge_complex(res, t, fx, fe, max_gap_h=3):
    """Join detections that belong to one complex flare (flux stays > 1σ above normal between peaks)."""
    if len(res) < 2:
        return res.assign(n_peaks=1, merged=False)
    base = pd.Series(fx).rolling(361, center=True, min_periods=50).median().values      # 12-hour baseline
    z = pd.Series((fx / base - 1) / (fe / base)).rolling(3, center=True, min_periods=1).mean().values
    res = res.sort_values('peak_time').reset_index(drop=True)
    groups = [[0]]
    for k in range(1, len(res)):
        a, b = res.peak_time[groups[-1][-1]], res.peak_time[k]
        between = (t > a) & (t < b)
        same = (b - a) * 24 <= max_gap_h and between.any() and np.nanmin(z[between]) > 1
        if same:
            groups[-1].append(k)
        else:
            groups.append([k])
    rows = []
    for g in groups:
        sub = res.loc[g]
        main = sub.loc[sub.peak_brightening_pct.idxmax()].copy()   # brighter peak; its window covers the flare
        main['flare_prob'] = sub.flare_prob.max()
        main['level'] = 'confirmed' if (sub.level == 'confirmed').any() else 'candidate'
        main['n_peaks'], main['merged'] = len(g), len(g) > 1
        rows.append(main)
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------- full run
def run(df, teff, radius, contam, model, prep, levels):
    t, fx, fe = df.time.values, df.flux.values, df.flux_err.values
    cadence = float(np.median(np.diff(t)) * 1440)
    info = dict(n_points=len(df), days=float(t.max() - t.min()), cadence_min=cadence,
                star_type=star_type(teff), warnings=check_inputs(teff, radius, contam, cadence))
    sigmas = sorted({parse_rule(lv['rule'])[1] for lv in levels.values() if isinstance(lv, dict)} - {0.0})
    c, info['n_windows'] = detect(t, fx, fe, teff, radius, model, prep, sigmas)
    cols = ['flare_id', 'peak_time', 'flare_prob', 'peak_brightening_pct', 'snr', 'low_snr',
            'energy_erg', 'log10_energy', 'type', 'level', 'merged']
    if len(c):
        c = apply_levels(c, levels)
    if len(c) == 0:
        return pd.DataFrame(columns=cols), info
    res = merge_complex(c, t, fx, fe)
    res['low_snr'] = res.snr < 5
    res['type'] = [TYPES[np.digitize(e, CUTS)] for e in res.log10_energy]
    res['energy_erg'] = [float(f'{10 ** e:.2e}') for e in res.log10_energy]
    res = res.sort_values('peak_time').reset_index(drop=True)
    res['flare_id'] = range(1, len(res) + 1)
    res = res[cols]
    return res.round({'peak_time': 4, 'flare_prob': 3, 'peak_brightening_pct': 3,
                      'snr': 1, 'log10_energy': 2}), info


# ---------------------------------------------------------------- hand-offs
def cutouts_for_c2(res, df, star_id='uploaded_star', half_window_d=0.15):
    """Flare cut-outs for Component 2 (CME): flare_id, star_id, time, flux, tpeak."""
    t, f = df.time.values, df.flux.values / np.median(df.flux.values)
    parts = []
    for r in res[res.level == 'confirmed'].itertuples():
        m = np.abs(t - r.peak_time) <= half_window_d
        parts.append(pd.DataFrame(dict(flare_id=r.flare_id, star_id=star_id, time=t[m], flux=f[m],
                                       tpeak=r.peak_time)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=['flare_id', 'star_id', 'time', 'flux', 'tpeak'])


def summary_for_c4(res, teff, radius):
    """Strongest confirmed flare for Component 4 (habitability)."""
    conf = res[res.level == 'confirmed']
    if len(conf) == 0:
        return dict(teff=teff, radius=radius, n_confirmed_flares=0, strongest_log10_energy=None,
                    strongest_type=None)
    top = conf.loc[conf.log10_energy.idxmax()]
    return dict(teff=teff, radius=radius, n_confirmed_flares=int(len(conf)),
                strongest_log10_energy=float(top.log10_energy), strongest_type=top.type,
                strongest_peak_time=float(top.peak_time))
