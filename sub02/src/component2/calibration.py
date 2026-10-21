"""Platt scaling of the raw model score."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Platt:
    """Logistic map from raw score to calibrated probability (monotone)."""

    def __init__(self, C: float = 1e6):
        self.C = C
        self.lr: LogisticRegression | None = None

    def fit(self, p_cal, y_cal) -> "Platt":
        self.lr = LogisticRegression(C=self.C).fit(logit(p_cal).reshape(-1, 1), np.asarray(y_cal))
        return self

    def __call__(self, p) -> np.ndarray:
        if self.lr is None:
            raise RuntimeError("Platt calibrator is not fitted")
        return self.lr.predict_proba(logit(p).reshape(-1, 1))[:, 1]
