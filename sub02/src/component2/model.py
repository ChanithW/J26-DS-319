"""Classifier factory and the shared preprocessing (prep -> median fill -> scale)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .features import prep


def make_model(y, seed: int = 0) -> XGBClassifier:
    """The research classifier (XGBoost, class-weighted)."""
    y = np.asarray(y)
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                         eval_metric="logloss", random_state=seed, n_jobs=1)


def pooled_stats(solar: pd.DataFrame, stellar: pd.DataFrame, cols: list[str]):
    """Pooled median and std over solar + stellar flares (after the log step)."""
    P = pd.concat([prep(solar, cols), prep(stellar, cols)], ignore_index=True)
    med = P.median()
    return med, P.fillna(med).std()                      # notebook: std of the median-filled pooled data


def to_scaled(df: pd.DataFrame, cols: list[str], med: pd.Series, sd0: pd.Series) -> np.ndarray:
    """prep -> fill with the stored pooled median -> divide by the stored pooled std."""
    sd = sd0.replace(0, 1.0)
    return (prep(df, cols).fillna(med) / sd).values.astype(float)
