"""Endpoints under /api/v1."""
from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile

from ..schemas import FlareInput, HealthResponse, ModelInfo, PredictRequest, RunListItem, RunResponse
from .csvio import CsvError, parse_long_csv
from .db import Run
from .errors import ApiError
from .service import build_run, export_run, load_run, load_samples

router = APIRouter(prefix="/api/v1")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _bundle(request: Request) -> dict:
    b = request.app.state.bundle
    if b is None:
        raise ApiError(503, "model_unavailable", "The model bundle could not be loaded.",
                       request.app.state.bundle_error)
    return b


def _check_limits(request: Request, records: list[dict]) -> None:
    s = request.app.state.settings
    if len(records) > s.max_flares_per_request:
        raise ApiError(413, "too_many_flares", f"At most {s.max_flares_per_request} flares per request.")
    for r in records:
        if len(r.get("time") or []) > s.max_points_per_flare:
            raise ApiError(413, "too_many_points",
                           f"Flare {r.get('flare_id')} has more than {s.max_points_per_flare} points.")


def _fill_fetched(request: Request, records: list[dict]) -> list[dict]:
    """Fetch the curve for records without one, only if the (off by default) flag is on."""
    if not request.app.state.settings.enable_lightkurve_fetch:
        return records
    from ..fetch import fetch_spoc_2min

    for r in records:
        if r.get("time") is None or r.get("flux") is None:
            try:
                t, f = fetch_spoc_2min(r["star_id"], r["tpeak"])
                r["time"], r["flux"] = t.tolist(), f.tolist()
            except Exception:
                pass                                     # stays invalid_input
    return records


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict:
    """Light check; does not run the model."""
    b = _bundle(request)
    return {"status": "ok", "model_version": b["version"]}


@router.get("/model-info", response_model=ModelInfo)
def model_info(request: Request) -> dict:
    b = _bundle(request)
    s = b["summary"]
    return {"bundle_version": b["version"], "created": b["created"], "data_hash": b["data_hash"],
            "training_summary": {k: v for k, v in s.items() if k not in ("limitations", "report")},
            "metrics": b.get("metrics") or s["report"], "limitations": s["limitations"]}


@router.get("/samples", response_model=list[FlareInput])
def samples(request: Request) -> list[dict]:
    """The 33 TESS sample flares in the request format of POST /predict."""
    out = load_samples(request.app.state.settings.sample_data_dir)
    if not out:
        raise ApiError(404, "samples_unavailable", "No sample curves were found (check SAMPLE_DATA_DIR).")
    return out


@router.post("/predict", response_model=RunResponse)
def predict(req: PredictRequest, request: Request) -> dict:
    bundle = _bundle(request)
    records = [f.model_dump() for f in req.flares]
    _check_limits(request, records)
    records = _fill_fetched(request, records)
    return build_run(bundle, records, req.save_run, req.run_name, request.app.state.session_factory,
                     request.app.state.settings.out_of_range_check)


@router.post("/predict/file", response_model=RunResponse)
async def predict_file(request: Request, file: UploadFile = File(...), save_run: bool = Form(False),
                       run_name: str | None = Form(None)) -> dict:
    bundle = _bundle(request)
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "file_too_large", "The file is larger than 50 MB.")
    try:
        records = parse_long_csv(raw)
    except CsvError as e:
        raise ApiError(422, "invalid_csv", str(e))
    _check_limits(request, records)
    return build_run(bundle, records, save_run, run_name, request.app.state.session_factory,
                     request.app.state.settings.out_of_range_check)


@router.get("/runs", response_model=list[RunListItem])
def list_runs(request: Request) -> list[dict]:
    with request.app.state.session_factory() as s:
        runs = s.query(Run).order_by(Run.created_at.desc()).all()
        out = []
        for r in runs:
            sm = json.loads(r.summary_json)
            out.append({"run_id": r.id, "name": r.name, "created_at": r.created_at.isoformat(timespec="seconds"),
                        "model_version": r.model_version, "n_flares": r.n_flares,
                        "n_eruptive": sm.get("n_eruptive", 0), "n_confident": sm.get("n_confident", 0)})
        return out


def _get_run(request: Request, run_id: str) -> dict:
    run = load_run(request.app.state.session_factory, run_id)
    if run is None:
        raise ApiError(404, "run_not_found", "That run does not exist.")
    return run


@router.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str, request: Request) -> dict:
    return _get_run(request, run_id)


@router.delete("/runs/{run_id}", status_code=204)
def delete_run(run_id: str, request: Request) -> Response:
    from .db import Prediction

    with request.app.state.session_factory() as s:
        run = s.get(Run, run_id)
        if run is None:
            raise ApiError(404, "run_not_found", "That run does not exist.")
        s.query(Prediction).filter(Prediction.run_id == run_id).delete()
        s.delete(run)
        s.commit()
    return Response(status_code=204)


@router.get("/runs/{run_id}/export")
def export(run_id: str, request: Request,
           format: str = Query("handoff_csv", pattern="^(handoff_csv|full_csv|json)$")) -> Response:
    body, media, name = export_run(_get_run(request, run_id), format)
    return Response(content=body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})
