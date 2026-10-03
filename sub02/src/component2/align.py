"""Solar-to-stellar feature alignment: none / diag / coral.

All inputs are preprocessed, median-filled and scaled by the pooled ``sd0``.
Alignment is applied to the solar training side only.
"""
from __future__ import annotations

import numpy as np


def _sqrtm(C: np.ndarray, inv: bool = False) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-9, None)
    p = -0.5 if inv else 0.5
    return (V * w ** p) @ V.T


def fit_alignment(Ls: np.ndarray, Lt: np.ndarray, method: str = "diag") -> dict:
    """Fit alignment constants from solar (Ls) and stellar (Lt) features."""
    if method == "none":
        return {"method": "none"}
    mu_t = Lt.mean(0)
    if method == "diag":
        return {"method": "diag", "mu_s": Ls.mean(0), "sd_s": Ls.std(0),
                "mu_t": mu_t, "sd_t": Lt.std(0) + 1e-6}
    if method == "coral":
        d = Ls.shape[1]
        Cs = np.cov(Ls, rowvar=False) + np.eye(d)
        Ct = np.cov(Lt, rowvar=False) + np.eye(d)
        return {"method": "coral", "mu_s": Ls.mean(0), "mu_t": mu_t,
                "W": _sqrtm(Cs, inv=True) @ _sqrtm(Ct)}
    raise ValueError(f"unknown alignment method: {method}")


def apply_alignment(Ls: np.ndarray, params: dict) -> np.ndarray:
    """Apply fitted alignment constants to solar features."""
    m = params["method"]
    if m == "none":
        return Ls.copy()
    if m == "diag":
        sd_s = np.where(params["sd_s"] == 0, 1.0, params["sd_s"])
        return (Ls - params["mu_s"]) / sd_s * params["sd_t"] + params["mu_t"]
    if m == "coral":
        return (Ls - params["mu_s"]) @ params["W"] + params["mu_t"]
    raise ValueError(f"unknown alignment method: {m}")
