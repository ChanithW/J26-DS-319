"""
Evaluate the deployed Sub-topic 3 model on the ORIGINAL held-out test set.

The split is not stored in the pickle, so it is re-created with the settings
used in training (20 % test, stratified by spectral_type, random_state=42) and
then VERIFIED: the test RMSE must match the saved training results
(10_point_metrics_test.csv: rate 0.388, energy 0.297). If it does not match,
the script stops instead of silently evaluating on a different split.

Metrics (overall and per spectral type), on the model scale
(log10 flares/day and log10 erg):
    RMSE, R2 (point model), pinball loss at q=0.10 / 0.90 (interval bounds),
    PICP (coverage of the 80 % interval) and MPIW (mean interval width).

Run from sub03/:   python scripts/evaluate.py
Outputs:           outputs/metrics_overall.csv, outputs/metrics_by_type.csv,
                   outputs/metrics.json, outputs/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subtopic3 import config  # noqa: E402
from subtopic3.predictor import FlarePredictor, prepare_input  # noqa: E402

# Test RMSE of the deployed models from training (10_point_metrics_test.csv).
EXPECTED_TEST_RMSE = {"log_flares_per_day": 0.388, "median_logE": 0.297}
RMSE_TOLERANCE = 0.0015  # values in that table are rounded to 3 decimals

# (observed column, prediction prefix in predict_log_frame output)
TARGETS = {"log_flares_per_day": "log_rate", "median_logE": "logE"}
TYPE_ORDER = ["F", "G", "K", "M"]


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    diff = y - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def compute_metrics(y, pt, lo, hi) -> dict:
    resid = y - pt
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "n": int(len(y)),
        "RMSE": float(np.sqrt(np.mean(resid ** 2))),
        "R2": float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "pinball_10": pinball(y, lo, 0.10),
        "pinball_90": pinball(y, hi, 0.90),
        "PICP": float(np.mean((y >= lo) & (y <= hi))),
        "MPIW": float(np.mean(hi - lo)),
    }


def reproduce_test_split(df: pd.DataFrame) -> pd.DataFrame:
    _, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["spectral_type"],
        random_state=config.SPLIT_RANDOM_STATE,
    )
    return test


def plot_pred_vs_obs(test: pd.DataFrame, preds: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    labels = {"log_flares_per_day": "log10 flare rate [flares/day]", "median_logE": "median log10 E [erg]"}
    colors = dict(zip(TYPE_ORDER, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))
    for ax, (target, p) in zip(axes, TARGETS.items()):
        y = test[target].to_numpy()
        for t in TYPE_ORDER:
            m = (preds["spectral_type"] == t).to_numpy()
            yerr = np.vstack([preds[p][m] - preds[f"{p}_lower"][m], preds[f"{p}_upper"][m] - preds[p][m]])
            ax.errorbar(y[m], preds[p][m], yerr=yerr, fmt="o", ms=3, alpha=0.5,
                        elinewidth=0.6, color=colors[t], label=t)
        lims = [min(y.min(), preds[f"{p}_lower"].min()), max(y.max(), preds[f"{p}_upper"].max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel(f"Observed {labels[target]}")
        ax.set_ylabel(f"Predicted {labels[target]}")
        ax.set_title(f"{target}: point + 80 % interval (test set)")
        ax.legend(title="Type")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_by_type(by_type: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["RMSE", "PICP", "MPIW"]):
        pivot = by_type.pivot(index="spectral_type", columns="target", values=metric).loc[TYPE_ORDER]
        pivot.plot.bar(ax=ax, rot=0)
        if metric == "PICP":
            ax.axhline(0.80, color="k", ls="--", lw=1, label="nominal 0.80")
            ax.set_ylim(0, 1)
        ax.set_title(f"{metric} by spectral type (test set)")
        ax.set_xlabel("Spectral type")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.DATA_PATH)
    test = reproduce_test_split(df).reset_index(drop=True)

    predictor = FlarePredictor()
    preds = predictor.predict_log_frame([prepare_input(r) for r in test.to_dict("records")])

    overall_rows, type_rows = [], []
    for target, p in TARGETS.items():
        y, pt = test[target].to_numpy(), preds[p].to_numpy()
        lo, hi = preds[f"{p}_lower"].to_numpy(), preds[f"{p}_upper"].to_numpy()
        overall_rows.append({"target": target, **compute_metrics(y, pt, lo, hi)})
        for t in TYPE_ORDER:
            m = (test["spectral_type"] == t).to_numpy()
            type_rows.append({"target": target, "spectral_type": t,
                              **compute_metrics(y[m], pt[m], lo[m], hi[m])})
    overall = pd.DataFrame(overall_rows)
    by_type = pd.DataFrame(type_rows)

    # ---- verify this is the original split before saving anything
    for _, row in overall.iterrows():
        expected = EXPECTED_TEST_RMSE[row["target"]]
        if abs(row["RMSE"] - expected) > RMSE_TOLERANCE:
            sys.exit(
                f"Split check FAILED for {row['target']}: RMSE {row['RMSE']:.4f} vs "
                f"{expected} from training. The original test split was not reproduced; "
                "no metrics were saved."
            )
    print(f"Split check passed: test RMSE matches training (n_test = {len(test)}).")

    overall.to_csv(config.OUTPUT_DIR / "metrics_overall.csv", index=False)
    by_type.to_csv(config.OUTPUT_DIR / "metrics_by_type.csv", index=False)
    (config.OUTPUT_DIR / "metrics.json").write_text(json.dumps({
        "split": {"test_size": config.TEST_SIZE, "stratify": "spectral_type",
                  "random_state": config.SPLIT_RANDOM_STATE, "n_test": len(test),
                  "verified_against": EXPECTED_TEST_RMSE},
        "interval_level": predictor.interval_level,
        "overall": overall_rows, "by_spectral_type": type_rows,
    }, indent=2))
    plot_pred_vs_obs(test, preds, config.OUTPUT_DIR / "pred_vs_obs_intervals.png")
    plot_by_type(by_type, config.OUTPUT_DIR / "metrics_by_type.png")

    pd.set_option("display.width", 140)
    print("\nOverall:\n", overall.round(3).to_string(index=False))
    print("\nBy spectral type:\n", by_type.round(3).to_string(index=False))
    print(f"\nSaved metrics and plots to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
