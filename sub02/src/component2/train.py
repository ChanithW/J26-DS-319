"""Build features, split, train, calibrate, bootstrap, conformal, and assemble the bundle.

Stellar flares are never training examples: they only feed the pooled statistics,
the alignment constants and the conformal weights model.
"""
from __future__ import annotations

import hashlib
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

from . import BUNDLE_VERSION
from .align import apply_alignment, fit_alignment
from .calibration import Platt
from .config import Config
from .conformal import WeightsModel, nonconformity, q_plain, q_weighted
from .data_io import (load_solar_labels, load_solar_raw, load_tess_features, solar_raw_path,
                      warn_unlabelled_solar_files)
from .evaluate import binary_metrics, conformal_report
from .features import FEATS13, shape_feats
from .model import make_model, pooled_stats, to_scaled
from .windowing import cut_window, to_2min

log = logging.getLogger(__name__)

LIMITATIONS = [
    "Trained on solar flares only; stellar flares have no CME labels, so accuracy on stars cannot be measured.",
    "Skill on the Sun is modest (TSS about 0.1-0.2, AUC about 0.6).",
    "The 5-95% range covers variability from the solar training data only, not the solar-to-stellar gap.",
    "The share of stellar flares called eruptive depends on the alignment choice; no stellar CME rate can be read from the outputs.",
    "The calibrated probability is a ranking score, not a trustworthy chance of a CME (Brier equals the base-rate baseline).",
    "Never present a stellar flare as 'a CME' or 'a detection'.",
]


def build_solar_features(data_dir: str | Path, cfg: Config) -> pd.DataFrame:
    """Features for every labelled solar flare (about 712 of 719 are expected to succeed)."""
    labels = load_solar_labels(data_dir)
    warn_unlabelled_solar_files(data_dir, labels)
    rs = np.random.RandomState(cfg.phase_seed)
    lo, hi = cfg.window_min
    rows = []
    for r in labels.itertuples(index=False):
        phase = rs.uniform(0, 2)                         # drawn in flare order, one per flare
        if not solar_raw_path(data_dir, r.flare_id).exists():
            log.warning("solar raw file missing for flare %s; skipped", r.flare_id)
            continue
        raw = load_solar_raw(data_dir, r.flare_id)
        t = (raw["timestamp"] - r.peak_time).dt.total_seconds().values / 60.0
        f = raw["xrsb"].values.astype(float)
        ok = (raw["xrsb_quality"].values == 0) & np.isfinite(f) & (f > 0)
        t, f = cut_window(t[ok], f[ok], lo, hi)
        tb, fb = to_2min(t, f, phase, cfg.min_bin_points)
        if len(tb) < cfg.min_points:
            continue
        feats = shape_feats(tb, fb)
        if feats is None:
            continue
        rows.append({"flare_id": r.flare_id, "ar_number": r.ar_number, "eruptive": r.eruptive, **feats})
    df = pd.DataFrame(rows)
    log.info("solar flares with features: %d of %d", len(df), len(labels))
    return df


def _data_hash(data_dir: Path) -> str:
    h = hashlib.sha256()
    for p in [data_dir / "solar_labels" / "labels_seed_standardized.csv", data_dir / "tess_sample_features2.csv"]:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _bootstrap(tr_idx, y, groups, X, cfg: Config) -> list:
    """Resample active regions of the training split with replacement; fit one model each."""
    rs = np.random.RandomState(cfg.bootstrap_seed)
    ars = np.unique(groups[tr_idx])
    by_ar = {a: tr_idx[groups[tr_idx] == a] for a in ars}
    models = []
    for b in range(cfg.n_bootstrap):
        pick = rs.choice(ars, size=len(ars), replace=True)
        idx = np.concatenate([by_ar[a] for a in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        models.append(make_model(y[idx], seed=b).fit(X[idx], y[idx]))
    return models


def train_from_tables(solar: pd.DataFrame, stellar: pd.DataFrame, cfg: Config, data_hash: str = "n/a"):
    """Train from a solar feature table and a stellar feature table; return (bundle, report)."""
    solar = solar.reset_index(drop=True)
    y = solar["eruptive"].values.astype(int)
    groups = solar["ar_number"].values

    med, sd0 = pooled_stats(solar, stellar, FEATS13)
    Ls = to_scaled(solar, FEATS13, med, sd0)
    Lt = to_scaled(stellar, FEATS13, med, sd0)
    align = fit_alignment(Ls, Lt, cfg.alignment)
    As = apply_alignment(Ls, align)

    folds = list(StratifiedGroupKFold(cfg.split_folds, shuffle=True, random_state=cfg.split_seed)
                 .split(As, y, groups))
    te, ca = folds[0][1], folds[1][1]
    tr = np.concatenate([f[1] for f in folds[2:]])
    for a, b in [(tr, te), (tr, ca), (ca, te)]:
        assert not set(groups[a]) & set(groups[b]), "active region split across partitions"

    model = make_model(y[tr], seed=0).fit(As[tr], y[tr])
    boots = _bootstrap(tr, y, groups, As, cfg)

    p_ca = model.predict_proba(As[ca])[:, 1]
    p_te = model.predict_proba(As[te])[:, 1]
    platt = Platt().fit(p_ca, y[ca])
    base = float(y[ca].mean())

    lo_q, hi_q = cfg.range_percentiles
    range_lo = np.percentile(As[tr], lo_q, axis=0)
    range_hi = np.percentile(As[tr], hi_q, axis=0)

    nc_ca = nonconformity(p_ca, y[ca])
    wm = WeightsModel().fit(As[ca], Lt)
    w_ca = wm(As[ca])
    q1 = q_plain(nc_ca[y[ca] == 1], cfg.alpha)
    q0 = q_plain(nc_ca[y[ca] == 0], cfg.alpha)
    conformal = {"nc_ca": nc_ca, "y_ca": y[ca], "w_ca": w_ca, "weights_model": wm, "q1": q1, "q0": q0,
                 "alpha": cfg.alpha}

    report = _report(solar, As, y, groups, te, ca, tr, model, p_te, platt, base, conformal, cfg, Lt, wm, w_ca)

    bundle = {
        "version": BUNDLE_VERSION, "model": model, "bootstrap_models": boots, "feats": FEATS13,
        "pooled_median": med, "sd0": sd0, "alignment": align, "platt": platt, "base": base,
        "range_lo": range_lo, "range_hi": range_hi, "conformal": conformal, "config": cfg.to_dict(),
        "data_hash": data_hash,
        "library_versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__,
                             "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {"n_solar": len(solar), "n_eruptive": int(y.sum()), "n_train": len(tr),
                    "n_cal": len(ca), "n_test": len(te), "n_bootstrap_models": len(boots),
                    "n_stellar_reference": len(stellar), "report": report, "limitations": LIMITATIONS},
    }
    return bundle, report


def _report(solar, As, y, groups, te, ca, tr, model, p_te, platt, base, conformal, cfg, Lt, wm, w_ca) -> dict:
    """Acceptance-check numbers (section 9 of the brief)."""
    cv_pred = np.zeros(len(y))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(As, y, groups):
        cv_pred[b] = make_model(y[a], seed=0).fit(As[a], y[a]).predict_proba(As[b])[:, 1]
    cv = binary_metrics(y, (cv_pred > 0.5).astype(int), cv_pred)
    test = binary_metrics(y[te], (p_te > 0.5).astype(int), p_te)

    # weighted conformal on the solar test flares
    nc, yc, w = conformal["nc_ca"], conformal["y_ca"], conformal["w_ca"]
    wt = wm(As[te])
    q1 = np.array([q_weighted(nc[yc == 1], w[yc == 1], x, cfg.alpha) for x in wt])
    q0 = np.array([q_weighted(nc[yc == 0], w[yc == 0], x, cfg.alpha) for x in wt])
    conf = conformal_report(y[te], p_te, q1, q0)
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    cp_te = platt(p_te)
    return {
        "n_solar_with_features": len(y), "n_eruptive": int(y.sum()), "n_confined": int((y == 0).sum()),
        "split": {"train": len(tr), "calibration": len(ca), "test": len(te)},
        "cv5_grouped": cv, "solar_test_raw_0.5": test, "solar_test_conformal_weighted": conf,
        "effective_cal_size": ess, "n_cal": len(ca), "base_rate_cal": base,
        "brier_test": float(brier_score_loss(y[te], cp_te)),
        "brier_test_base_rate_baseline": float(brier_score_loss(y[te], np.full(len(te), base))),
        "always_confined_accuracy": float(1 - y[te].mean()),
        "test_tss_label_rule_platt_base": binary_metrics(y[te], (cp_te >= base).astype(int), cp_te)["tss"],
    }


def evaluate_bundle(bundle: dict, data_dir: str | Path) -> dict:
    """Re-run the held-out solar test checks with a stored bundle (same split as training)."""
    cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in bundle["config"].items()})
    solar = build_solar_features(data_dir, cfg).reset_index(drop=True)
    y = solar["eruptive"].values.astype(int)
    groups = solar["ar_number"].values
    L = to_scaled(solar, bundle["feats"], bundle["pooled_median"], bundle["sd0"])
    As = apply_alignment(L, bundle["alignment"])
    folds = list(StratifiedGroupKFold(cfg.split_folds, shuffle=True, random_state=cfg.split_seed)
                 .split(As, y, groups))
    te = folds[0][1]
    p = bundle["model"].predict_proba(As[te])[:, 1]
    cp = bundle["platt"](p)
    cf = bundle["conformal"]
    wt = cf["weights_model"](As[te])
    nc, yc, w = cf["nc_ca"], cf["y_ca"], cf["w_ca"]
    q1 = np.array([q_weighted(nc[yc == 1], w[yc == 1], x, cfg.alpha) for x in wt])
    q0 = np.array([q_weighted(nc[yc == 0], w[yc == 0], x, cfg.alpha) for x in wt])
    return {"n_test": len(te), "raw_0.5": binary_metrics(y[te], (p > 0.5).astype(int), p),
            "platt_base_rule": binary_metrics(y[te], (cp >= bundle["base"]).astype(int), cp),
            "conformal": conformal_report(y[te], p, q1, q0),
            "brier": float(brier_score_loss(y[te], cp))}


def train(data_dir: str | Path, cfg: Config):
    """Full training from the data folder; returns (bundle, report)."""
    data_dir = Path(data_dir)
    solar = build_solar_features(data_dir, cfg)
    stellar = load_tess_features(data_dir)
    return train_from_tables(solar, stellar, cfg, _data_hash(data_dir))
