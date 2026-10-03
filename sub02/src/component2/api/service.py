"""Building run responses, saving runs and exports."""
from __future__ import annotations

import io
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..handoff import HANDOFF_COLUMNS
from ..predict import OUTPUT_COLUMNS, predict_records
from .db import Prediction, Run


def _clean(v):
    """Make a value JSON-safe (NaN -> None, numpy -> python)."""
    if hasattr(v, "item") and not isinstance(v, (list, dict)):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def run_prediction(bundle: dict, records: list[dict]) -> tuple[list[dict], dict]:
    """Run the model; returns (results, summary). One bad flare never fails the batch."""
    df = predict_records(bundle, records, details=True)
    results = [_clean(r) for r in df.to_dict(orient="records")]
    for r in results:
        r["warnings"] = r.get("warnings") or []
    ok = [r for r in results if r["status"] == "ok"]
    summary = {
        "n_flares": len(results), "n_ok": len(ok),
        "n_eruptive": sum(r["predicted_label"] == "eruptive" for r in ok),
        "n_confined": sum(r["predicted_label"] == "confined" for r in ok),
        "n_confident": sum(r["reliability"] == "confident" for r in ok),
        "n_uncertain": sum(r["reliability"] == "uncertain" for r in ok),
    }
    return results, summary


def build_run(bundle: dict, records: list[dict], save: bool, name: str | None, session_factory) -> dict:
    results, summary = run_prediction(bundle, records)
    now = datetime.now(timezone.utc)
    run_id = None
    if save:
        run_id = str(uuid.uuid4())
        with session_factory() as s:
            s.add(Run(id=run_id, name=name, created_at=now, model_version=bundle["version"],
                      n_flares=len(results), summary_json=json.dumps(summary)))
            s.add_all(Prediction(run_id=run_id, flare_id=r["flare_id"], star_id=r["star_id"],
                                 status=r["status"], result_json=json.dumps(r)) for r in results)
            s.commit()
    return {"run_id": run_id, "name": name, "model_version": bundle["version"],
            "created_at": now.isoformat(timespec="seconds"), "summary": summary, "results": results}


def load_run(session_factory, run_id: str) -> dict | None:
    with session_factory() as s:
        run = s.get(Run, run_id)
        if run is None:
            return None
        rows = s.query(Prediction).filter(Prediction.run_id == run_id).order_by(Prediction.id).all()
        return {"run_id": run.id, "name": run.name, "model_version": run.model_version,
                "created_at": run.created_at.isoformat(timespec="seconds"),
                "summary": json.loads(run.summary_json), "results": [json.loads(r.result_json) for r in rows]}


def export_run(run: dict, fmt: str) -> tuple[str, str, str]:
    """Return (body, media_type, filename)."""
    stem = f"run_{(run['run_id'] or 'unsaved')[:8]}"
    if fmt == "json":
        return json.dumps(run, indent=2), "application/json", f"{stem}.json"
    df = pd.DataFrame([{k: r.get(k) for k in OUTPUT_COLUMNS} for r in run["results"]])
    if fmt == "handoff_csv":
        df = df[df["status"] == "ok"][HANDOFF_COLUMNS]
    else:
        df["warnings"] = df["warnings"].map(lambda w: ";".join(w) if isinstance(w, list) else "")
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")        # same bytes on every platform
    return buf.getvalue(), "text/csv", f"{stem}_{fmt}.csv"


def load_samples(sample_dir: str | Path) -> list[dict]:
    """The 33 TESS sample flares in request format (tpeak = 0, time = tmin / 1440)."""
    out = []
    for p in sorted(Path(sample_dir).glob("tic_*.csv")):
        tic = int(p.stem.split("_", 1)[1])
        df = pd.read_csv(p)
        out.append({"flare_id": f"TESS_{tic}", "star_id": tic, "tpeak": 0.0,
                    "time": (df["tmin"] / 1440.0).tolist(), "flux": df["flux"].tolist()})
    return out
