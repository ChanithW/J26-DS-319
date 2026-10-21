import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from component2.api import create_app
from component2.api.dump_openapi import OPENAPI_PATH, build_schema
from component2.api.settings import Settings
from component2.bundle import save_bundle
from tests.test_pipeline import curve

P = "/api/v1"


@pytest.fixture()
def client(bundle, tmp_path):
    bpath = tmp_path / "b.joblib"
    save_bundle(bundle, bpath)
    samples = tmp_path / "tess_raw"
    samples.mkdir()
    for tic in (111, 222):
        c = curve()
        pd.DataFrame({"tmin": (np.array(c["time"]) - c["tpeak"]) * 1440, "flux": c["flux"]}).to_csv(
            samples / f"tic_{tic}.csv", index=False)
    s = Settings(component2_bundle_path=str(bpath), database_url=f"sqlite:///{tmp_path / 'runs.db'}",
                 sample_data_dir=str(samples), max_flares_per_request=5, max_points_per_flare=200)
    with TestClient(create_app(s)) as c:
        yield c


def long_csv(flares):
    rows = []
    for fid, c in flares:
        rows += [{"flare_id": fid, "star_id": 1, "tpeak": c["tpeak"], "time": t, "flux": f}
                 for t, f in zip(c["time"], c["flux"])]
    return pd.DataFrame(rows).to_csv(index=False).encode()


def test_health_and_model_info(client):
    assert client.get(f"{P}/health").json()["status"] == "ok"
    info = client.get(f"{P}/model-info").json()
    assert info["limitations"] and "metrics" in info and info["bundle_version"]
    assert "cv5_grouped" in info["metrics"]


def test_health_503_when_bundle_missing(tmp_path):
    s = Settings(component2_bundle_path=str(tmp_path / "nope.joblib"), database_url="sqlite://")
    with TestClient(create_app(s)) as c:
        r = c.get(f"{P}/health")
        assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


def test_samples(client):
    r = client.get(f"{P}/samples").json()
    assert len(r) == 2 and r[0]["flare_id"] == "TESS_111" and r[0]["tpeak"] == 0.0


def test_predict_one_bad_flare_returns_all_rows(client):
    good, flat = curve(), {**curve(), "flare_id": "flat", "flux": [5.0] * 55}
    r = client.post(f"{P}/predict", json={"flares": [good, flat, {**good, "flare_id": "G2"}]})
    assert r.status_code == 200
    body = r.json()
    assert [x["status"] for x in body["results"]] == ["ok", "no_peak", "ok"]
    assert body["summary"]["n_flares"] == 3 and body["summary"]["n_ok"] == 2
    ok = body["results"][0]
    assert ok["curve"]["g"] and len(ok["features"]) == 13
    assert body["results"][1]["curve"] is None and body["results"][1]["features"] is None
    assert body["run_id"] is None


def test_validation_and_limits_use_error_shape(client):
    r = client.post(f"{P}/predict", json={"flares": [{"flare_id": "x"}]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    many = [{**curve(), "flare_id": f"F{i}"} for i in range(6)]
    r = client.post(f"{P}/predict", json={"flares": many})
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_many_flares"
    big = {**curve(), "time": list(range(300)), "flux": [1.0] * 300}
    assert client.post(f"{P}/predict", json={"flares": [big]}).status_code == 413


def test_csv_upload_ok_and_errors(client):
    r = client.post(f"{P}/predict/file", files={"file": ("a.csv", long_csv([("A", curve()), ("B", curve())]))})
    assert r.status_code == 200 and r.json()["summary"]["n_flares"] == 2
    bad = client.post(f"{P}/predict/file", files={"file": ("a.csv", b"flare_id,time\nA,1\n")})
    assert bad.status_code == 422 and "Missing column" in bad.json()["error"]["message"]
    nonnum = long_csv([("A", curve())]).replace(b"10.0", b"abc", 1)
    assert client.post(f"{P}/predict/file", files={"file": ("a.csv", nonnum)}).status_code == 422
    one = b"flare_id,star_id,tpeak,time,flux\nA,1,100,100.0,1.0\n"
    assert "fewer than 2" in client.post(f"{P}/predict/file", files={"file": ("a.csv", one)}).json()["error"]["message"]
    toobig = long_csv([("A", {**curve(), "time": curve()["time"] * 4, "flux": curve()["flux"] * 4})])
    assert client.post(f"{P}/predict/file", files={"file": ("a.csv", toobig)}).status_code == 413


def test_saved_run_read_export_delete(client):
    r = client.post(f"{P}/predict", json={"flares": [curve(), {**curve(), "flare_id": "bad", "flux": [5.0] * 55}],
                                         "save_run": True, "run_name": "t1"}).json()
    rid = r["run_id"]
    assert rid
    assert client.get(f"{P}/runs").json()[0]["name"] == "t1"
    got = client.get(f"{P}/runs/{rid}").json()
    assert len(got["results"]) == 2 and got["summary"] == r["summary"]
    ho = client.get(f"{P}/runs/{rid}/export?format=handoff_csv").text.splitlines()
    assert ho[0].startswith("flare_id,star_id,calibrated_probability,confidence_lower,confidence_upper,predicted_label,reliability")
    assert len(ho) == 2                                  # non-ok row left out of the hand-off
    full = pd.read_csv(io.StringIO(client.get(f"{P}/runs/{rid}/export?format=full_csv").text))
    assert len(full) == 2 and "conformal_set" in full.columns
    assert json.loads(client.get(f"{P}/runs/{rid}/export?format=json").text)["run_id"] == rid
    assert client.delete(f"{P}/runs/{rid}").status_code == 204
    assert client.get(f"{P}/runs/{rid}").status_code == 404
    assert client.get(f"{P}/runs/{rid}").json()["error"]["code"] == "run_not_found"


def test_openapi_json_is_committed_and_current():
    assert OPENAPI_PATH.exists(), "run: python -m component2.api.dump_openapi"
    assert json.loads(OPENAPI_PATH.read_text(encoding="utf-8")) == json.loads(json.dumps(build_schema()))
