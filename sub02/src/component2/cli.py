"""Command line interface: component2 train | predict | evaluate | info | refit-alignment."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import BundleError, load_bundle, save_bundle
from .config import load_config
from .data_io import DataError, load_tess_features
from .handoff import write_handoff
from .predict import predict_records

log = logging.getLogger("component2")


def _fmt_report(report: dict) -> str:
    return json.dumps(report, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


def _train(args) -> int:
    from .train import build_solar_features, train_from_tables, _data_hash
    cfg = load_config(args.config)
    data = Path(args.data)
    solar = build_solar_features(data, cfg)
    stellar = pd.read_csv(args.stellar) if getattr(args, "stellar", None) else load_tess_features(data)
    bundle, report = train_from_tables(solar, stellar, cfg, _data_hash(data))
    save_bundle(bundle, args.out)
    Path(args.out).with_name("metrics.json").write_text(_fmt_report(bundle["metrics"]), encoding="utf-8")
    print(_fmt_report(report))
    print(f"bundle written to {args.out}")
    return 0


def _predict(args) -> int:
    bundle = load_bundle(args.bundle)
    with open(args.input, encoding="utf-8") as fh:
        data = json.load(fh)
    records = data["flares"] if isinstance(data, dict) else data
    pred = predict_records(bundle, records)
    pred.assign(warnings=pred["warnings"].map(lambda w: ";".join(w) if isinstance(w, list) else "")) \
        .to_csv(args.out, index=False)
    if args.handoff:
        write_handoff(pred, args.handoff)
    print(f"{len(pred)} rows written to {args.out} ({(pred['status'] == 'ok').sum()} ok)")
    return 0


def _evaluate(args) -> int:
    from .train import evaluate_bundle

    bundle = load_bundle(args.bundle)
    print(_fmt_report(evaluate_bundle(bundle, args.data)))
    return 0


def _info(args) -> int:
    b = load_bundle(args.bundle)
    print(f"version: {b['version']}\ncreated: {b['created']}\ndata hash: {b['data_hash']}")
    print("libraries:", b["library_versions"])
    s = dict(b["summary"])
    lim = s.pop("limitations")
    print("training summary:\n" + _fmt_report(s))
    print("limitations:")
    for line in lim:
        print(" -", line)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="component2", description="CME occurrence prediction for stellar flares")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="rebuild everything and print the evaluation report")
    t.add_argument("--data", default="data/")
    t.add_argument("--out", default="model_bundle_v3.joblib")
    t.add_argument("--config", default=None)
    t.add_argument("--stellar", default=None, help="stellar feature CSV (default: data/tess_sample_features2.csv)")
    t.set_defaults(fn=_train)

    r = sub.add_parser("refit-alignment", help="retrain with a larger stellar feature table")
    r.add_argument("--data", default="data/")
    r.add_argument("--out", default="model_bundle_v3.joblib")
    r.add_argument("--config", default=None)
    r.add_argument("--stellar", required=True)
    r.set_defaults(fn=_train)

    p = sub.add_parser("predict", help="batch prediction from a JSON file of flares")
    p.add_argument("--bundle", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--handoff", default=None, help="also write the Sub-topic 4 hand-off CSV")
    p.set_defaults(fn=_predict)

    e = sub.add_parser("evaluate", help="re-run the solar test-set checks")
    e.add_argument("--bundle", required=True)
    e.add_argument("--data", default="data/")
    e.set_defaults(fn=_evaluate)

    i = sub.add_parser("info", help="print bundle version, training summary and limitations")
    i.add_argument("--bundle", required=True)
    i.set_defaults(fn=_info)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.fn(args)
    except (DataError, BundleError, FileNotFoundError) as e_:
        print(f"error: {e_}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
