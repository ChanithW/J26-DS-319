"""Batch prediction for stellar flares (behaviour follows the notebook's predict_new)."""
from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .conformal import q_weighted, set_label
from .features import FEATS13, plot_curve, shape_feats
from .model import to_scaled
from .windowing import stellar_window

log = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["flare_id", "star_id", "status", "calibrated_probability", "confidence_lower",
                  "confidence_upper", "predicted_label", "reliability", "features_missing",
                  "out_of_range", "warnings", "raw_score", "conformal_set"]


def predict_features(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Score a table of flare features (columns: flare_id, star_id and the 13 features)."""
    cols = bundle["feats"]
    cfg = bundle["config"]
    missing = [c for c in ["flare_id", "star_id"] + cols if c not in df.columns]
    if missing:
        raise ValueError("missing input columns: %s" % missing)
    df = df.reset_index(drop=True)
    nmiss = df[cols].isna().sum(axis=1).values
    L = to_scaled(df, cols, bundle["pooled_median"], bundle["sd0"])
    p = bundle["model"].predict_proba(L)[:, 1]
    B = np.array([b.predict_proba(L)[:, 1] for b in bundle["bootstrap_models"]])
    lo_, hi_ = np.percentile(B, 5, axis=0), np.percentile(B, 95, axis=0)

    cf = bundle["conformal"]
    nc, yc, wc, alpha = cf["nc_ca"], cf["y_ca"], cf["w_ca"], cf["alpha"]
    sets = []
    for i in range(len(L)):
        if cfg["conformal_weighted"]:
            wt = float(cf["weights_model"](L[i:i + 1])[0])
            qa = q_weighted(nc[yc == 1], wc[yc == 1], wt, alpha)
            qb = q_weighted(nc[yc == 0], wc[yc == 0], wt, alpha)
        else:
            qa, qb = cf["q1"], cf["q0"]
        sets.append((bool((1 - p[i]) <= qa), bool(p[i] <= qb)))

    out_of_range = ((L < bundle["range_lo"]) | (L > bundle["range_hi"])).any(axis=1)
    platt = bundle["platt"]
    cp = platt(p)
    if cfg["label_rule"] == "raw_0.5":
        eruptive = p > 0.5
    else:
        eruptive = cp >= bundle["base"]

    warnings_, reliab = [], []
    for i, (ia, ib) in enumerate(sets):
        w = []
        ok = ia ^ ib
        if nmiss[i] > cfg["max_missing_features"]:
            w.append("many_features_missing")
            ok = False
        if out_of_range[i]:
            w.append("out_of_range")
            ok = False
        warnings_.append(w)
        reliab.append("confident" if ok else "uncertain")

    return pd.DataFrame({
        "flare_id": df["flare_id"].values, "star_id": df["star_id"].values, "status": "ok",
        "calibrated_probability": cp.round(3), "confidence_lower": platt(lo_).round(3),
        "confidence_upper": platt(hi_).round(3),
        "predicted_label": np.where(eruptive, "eruptive", "confined"),
        "reliability": reliab, "features_missing": nmiss, "out_of_range": out_of_range,
        "warnings": warnings_, "raw_score": p.round(4),
        "conformal_set": [set_label(a, b) for a, b in sets],
    })[OUTPUT_COLUMNS]


def _empty_row(rec: dict, status: str) -> dict:
    row = {c: None for c in OUTPUT_COLUMNS}
    row.update(flare_id=rec.get("flare_id"), star_id=rec.get("star_id"), status=status,
               warnings=[], features_missing=None, out_of_range=None)
    return row


def predict_records(bundle: dict, records: Iterable[dict[str, Any]], details: bool = False) -> pd.DataFrame:
    """Predict from raw light curves. One bad flare never fails the batch.

    Each record needs flare_id, star_id, tpeak, time (days, BTJD) and flux.
    With ``details=True`` two extra columns are added: ``features`` (the 13 features,
    None if missing) and ``curve`` (normalised curve for plotting).
    """
    cfg = bundle["config"]
    lo, hi = cfg["window_min"]
    rows: list[dict | None] = []
    feat_rows: list[dict] = []
    feat_pos: list[int] = []
    extras: dict[int, dict] = {}
    for rec in records:
        try:
            time, flux = np.asarray(rec["time"], float), np.asarray(rec["flux"], float)
            tpeak = float(rec["tpeak"])
            if time.ndim != 1 or time.shape != flux.shape or not np.isfinite(tpeak):
                raise ValueError("time/flux shape mismatch")
            if rec.get("flare_id") is None or rec.get("star_id") is None:
                raise ValueError("flare_id and star_id are required")
        except (KeyError, TypeError, ValueError) as e:
            log.warning("invalid input for %s: %s", rec.get("flare_id") if isinstance(rec, dict) else "?", e)
            rows.append(_empty_row(rec if isinstance(rec, dict) else {}, "invalid_input"))
            continue
        tm, f = stellar_window(time, flux, tpeak, lo, hi)
        if len(tm) < cfg["min_points"]:
            rows.append(_empty_row(rec, "too_few_points"))
            continue
        feats = shape_feats(tm, f)
        if feats is None:
            rows.append(_empty_row(rec, "no_peak"))
            continue
        extras[len(rows)] = {
            "features": {c: (None if not np.isfinite(feats[c]) else float(feats[c])) for c in FEATS13},
            "curve": plot_curve(tm, f)}
        feat_pos.append(len(rows))
        rows.append(None)
        feat_rows.append({"flare_id": rec["flare_id"], "star_id": rec["star_id"], **feats})
    if feat_rows:
        scored = predict_features(bundle, pd.DataFrame(feat_rows))
        for pos, (_, r) in zip(feat_pos, scored.iterrows()):
            rows[pos] = r.to_dict()
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if details:
        out["features"] = [extras.get(i, {}).get("features") for i in range(len(rows))]
        out["curve"] = [extras.get(i, {}).get("curve") for i in range(len(rows))]
    return out
