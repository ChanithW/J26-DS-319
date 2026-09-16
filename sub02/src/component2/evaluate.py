"""Evaluation metrics for the solar checks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def binary_metrics(y, pred, score=None) -> dict:
    """Accuracy, hit rate, clear rate, TSS, balanced accuracy and AUC."""
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    hit = tp / max((y == 1).sum(), 1)
    clear = tn / max((y == 0).sum(), 1)
    out = {"accuracy": (tp + tn) / len(y), "hit_rate": hit, "clear_rate": clear,
           "tss": hit + clear - 1, "balanced_accuracy": (hit + clear) / 2}
    if score is not None and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
    return out


def conformal_report(y, p, q1: np.ndarray | float, q0: np.ndarray | float) -> dict:
    """Coverage per class and the share/accuracy of single-label answers."""
    y, p = np.asarray(y), np.asarray(p, float)
    ia = (1 - p) <= q1
    ib = p <= q0
    single = ia ^ ib
    pred = np.where(ia, 1, 0)
    return {"coverage_eruptive": float(ia[y == 1].mean()), "coverage_confined": float(ib[y == 0].mean()),
            "single_label_share": float(single.mean()), "n_single": int(single.sum()),
            "single_label_accuracy": float((pred[single] == y[single]).mean()) if single.any() else float("nan")}


def brier(y, p) -> dict:
    y = np.asarray(y)
    return {"brier": float(brier_score_loss(y, p)), "brier_base_rate": float(brier_score_loss(y, np.full(len(y), y.mean())))}
