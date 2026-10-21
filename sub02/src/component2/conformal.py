"""Class-conditional split conformal prediction, plain and covariate-shift weighted."""
from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression


def nonconformity(p, y) -> np.ndarray:
    """1 - p for eruptive (y=1), p for confined (y=0)."""
    p, y = np.asarray(p, float), np.asarray(y)
    return np.where(y == 1, 1 - p, p)


def q_plain(nc, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score (inf if that index exceeds n)."""
    nc = np.sort(np.asarray(nc, float))
    n = len(nc)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(nc[k - 1]) if 1 <= k <= n else float("inf")


def q_weighted(nc, w, w_test: float, alpha: float) -> float:
    """Smallest score whose cumulative weight share first reaches 1 - alpha."""
    nc, w = np.asarray(nc, float), np.asarray(w, float)
    if len(nc) == 0:
        return float("inf")
    o = np.argsort(nc, kind="stable")
    cum = np.cumsum(w[o]) / (w.sum() + w_test)
    idx = np.where(cum >= 1 - alpha)[0]
    return float(nc[o][idx[0]]) if len(idx) else float("inf")


def in_sets(p, q1, q0):
    """Return (eruptive_in_set, confined_in_set)."""
    return (1 - np.asarray(p, float)) <= q1, np.asarray(p, float) <= q0


def set_label(ia: bool, ib: bool) -> str:
    """Name of the conformal set: E, C, E+C or none."""
    if ia and ib:
        return "E+C"
    if ia:
        return "E"
    if ib:
        return "C"
    return "none"


class WeightsModel:
    """Domain classifier giving each solar calibration flare a stellar-likeness weight."""

    def __init__(self, n_cal: int = 0, n_star: int = 1):
        self.n_cal, self.n_star = n_cal, n_star
        self.lr: LogisticRegression | None = None

    def fit(self, L_cal_aligned: np.ndarray, L_star: np.ndarray) -> "WeightsModel":
        X = np.vstack([L_cal_aligned, L_star])
        y = np.r_[np.zeros(len(L_cal_aligned)), np.ones(len(L_star))]
        self.n_cal, self.n_star = len(L_cal_aligned), len(L_star)
        self.lr = LogisticRegression(C=0.5, max_iter=1000).fit(X, y)
        return self

    def __call__(self, L: np.ndarray) -> np.ndarray:
        if self.lr is None:
            raise RuntimeError("weights model is not fitted")
        p = np.clip(self.lr.predict_proba(L)[:, 1], 1e-9, 1 - 1e-9)
        return np.clip(p / (1 - p) * self.n_cal / self.n_star, 0.05, 20.0)
