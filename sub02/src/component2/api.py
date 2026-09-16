"""FastAPI app wrapping prediction. Set COMPONENT2_BUNDLE to the bundle path."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from . import __version__
from .bundle import load_bundle
from .predict import predict_records
from .schemas import FlareResult, PredictRequest

app = FastAPI(title="Component 2: CME occurrence prediction", version=__version__)
_state: dict = {}


def get_bundle() -> dict:
    if "bundle" not in _state:
        path = os.environ.get("COMPONENT2_BUNDLE", "model_bundle_v3.joblib")
        if not Path(path).exists():
            raise HTTPException(503, f"model bundle not found: {path}")
        _state["bundle"] = load_bundle(path)
    return _state["bundle"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    b = get_bundle()
    s = b["summary"]
    return {"bundle_version": b["version"], "created": b["created"], "data_hash": b["data_hash"],
            "training_summary": {k: v for k, v in s.items() if k != "limitations"},
            "limitations": s["limitations"]}


@app.post("/predict", response_model=list[FlareResult])
def predict(req: PredictRequest) -> list[dict]:
    pred = predict_records(get_bundle(), [f.model_dump() for f in req.flares])
    out = []
    for r in pred.to_dict(orient="records"):
        out.append({k: (None if v is None or (isinstance(v, float) and v != v) else v) for k, v in r.items()})
    return out
