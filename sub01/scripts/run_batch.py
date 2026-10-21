"""Batch mode: analyse many stars and write CSV files for the other components.

Input: a CSV with a column "tic" (and optionally "sector"), for example the
planet-host list from the NASA Exoplanet Archive.

Output (in reports/runs/ by default):
    c1_all_flares.csv     one row per flare, all stars
    c1_star_summary.csv   one row per star: flare counts per tier, strongest tier
    c1_failed.csv         stars that could not be analysed, and why

Example (run from the repo root):
    python scripts/run_batch.py --input data/host_stars.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flare_cnn.config import load_config, load_star_scaling, load_threshold   # noqa: E402
from flare_cnn.detect import analyze_star  # noqa: E402
from flare_cnn.export import flares_table, save_csv, star_summary   # noqa: E402
from flare_cnn.model import load_model     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Run Component 1 on a list of TESS stars.")
    ap.add_argument("--input", required=True, help="CSV with a 'tic' column (optional 'sector')")
    ap.add_argument("--out", default=None, help="output folder (default from config.yaml)")
    args = ap.parse_args()

    cfg = load_config()
    mean, std = load_star_scaling(cfg)
    threshold = load_threshold(cfg)
    model = load_model(cfg["paths"]["model"])
    out = Path(args.out or cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    stars = pd.read_csv(args.input)
    if "tic" not in stars.columns:
        raise SystemExit("Input CSV needs a column called 'tic'")

    flares, summaries, failed = [], [], []
    for k, row in enumerate(stars.itertuples(index=False), 1):
        sector = getattr(row, "sector", None)
        sector = None if pd.isna(sector) else int(sector)
        try:
            r = analyze_star(model, row.tic, cfg, mean, std, threshold, sector=sector)
            flares.append(flares_table(r))
            summaries.append(star_summary(r))
            print(f"[{k}/{len(stars)}] TIC {r['tic']} S{r['sector']}: {len(r['events'])} flares")
        except Exception as e:     # one bad star must not stop the whole batch
            failed.append({"tic": row.tic, "sector": sector, "error": str(e)})
            print(f"[{k}/{len(stars)}] TIC {row.tic}: FAILED ({e})")

        # save after every star so nothing is lost if Colab disconnects
        if flares:
            save_csv(pd.concat(flares, ignore_index=True), out / "c1_all_flares.csv")
        save_csv(pd.DataFrame(summaries), out / "c1_star_summary.csv")
        save_csv(pd.DataFrame(failed, columns=["tic", "sector", "error"]), out / "c1_failed.csv")

    print(f"\nDone: {len(summaries)} stars analysed, {len(failed)} failed. Files in {out}")


if __name__ == "__main__":
    main()
