"""Analyse one star from the command line (no web app).

Examples (run from the repo root):
    python scripts/analyze_star.py 441420236 --sector 27
    python scripts/analyze_star.py "TIC 388857263" --teff 3050
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flare_cnn.config import load_config, load_star_scaling, load_threshold   # noqa: E402
from flare_cnn.detect import analyze_star  # noqa: E402
from flare_cnn.export import flares_table, save_csv, star_summary   # noqa: E402
from flare_cnn.model import load_model     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Detect flares and energy tiers for one TESS star.")
    ap.add_argument("tic", help="TIC ID, e.g. 441420236")
    ap.add_argument("--sector", type=int, default=None, help="TESS sector (default: first available)")
    ap.add_argument("--teff", type=float, default=None, help="Teff in K, if you know it")
    args = ap.parse_args()

    cfg = load_config()
    mean, std = load_star_scaling(cfg)
    model = load_model(cfg["paths"]["model"])
    r = analyze_star(model, args.tic, cfg, mean, std, load_threshold(cfg), args.sector, args.teff)

    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = save_csv(flares_table(r), out / f"c1_flares_TIC{r['tic']}_S{r['sector']}.csv")

    s = star_summary(r)
    print(f"TIC {r['tic']} sector {r['sector']} ({s['star_type']} star): {s['n_flares']} flares, "
          f"{s['n_weak']} weak | C {s['n_C']}  M {s['n_M']}  X {s['n_X']}  SF {s['n_Superflare']} | "
          f"catalogue recovered {s['catalogue_recovered']}/{s['catalogue_flares']} | {r['seconds']} s")
    print("Saved:", path)


if __name__ == "__main__":
    main()
