"""
Predict for every star in the dataset and write a static table for Sub-topic 4.

Columns follow the Sub-topic 3 output contract (subtopic3/contracts.py):
    KIC, spectral_type, pred_flare_rate, pred_flare_rate_lower, pred_flare_rate_upper,
    pred_logE, pred_logE_lower, pred_logE_upper, detection_bias_flag, interval_level

Run from sub03/:   python scripts/batch_export.py [--input other.csv] [--output path.csv]
Default output:    outputs/subtopic3_predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subtopic3 import config  # noqa: E402
from subtopic3.predictor import FlarePredictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=config.DATA_PATH)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "subtopic3_predictions.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = FlarePredictor().predict_frame(df)

    n_err = int(result["error"].notna().sum())
    if n_err == 0:
        result = result.drop(columns="error")  # keep the table exactly on-contract
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} rows ({n_err} rejected) to {args.output}")


if __name__ == "__main__":
    main()
