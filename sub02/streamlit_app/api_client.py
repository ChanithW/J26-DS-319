"""Thin client for the Component 2 API. The UI talks only to this API (never to the model code)."""
from __future__ import annotations

import os

import requests

# One environment variable for the API address.
BASE = os.environ.get("COMPONENT2_API_URL", "http://localhost:8000").rstrip("/")
API = f"{BASE}/api/v1"
TIMEOUT = 180


class ApiError(Exception):
    """An error from the API, with the message from its {"error": {...}} body."""

    def __init__(self, message: str, status: int = 0, code: str = "http_error"):
        super().__init__(message)
        self.message, self.status, self.code = message, status, code

    @property
    def unreachable(self) -> bool:
        return self.status == 0


def _request(method: str, path: str, **kw):
    try:
        res = requests.request(method, f"{API}{path}", timeout=TIMEOUT, **kw)
    except requests.RequestException:
        raise ApiError("Cannot reach the server. Check that the API is running and try again.", 0, "network")
    if not res.ok:
        try:
            err = res.json()["error"]
            raise ApiError(err.get("message") or f"The server answered with an error ({res.status_code}).",
                           res.status_code, err.get("code") or "http_error")
        except (ValueError, KeyError, TypeError):
            raise ApiError(f"The server answered with an error ({res.status_code}).", res.status_code)
    return None if res.status_code == 204 else res.json()


def model_info() -> dict:
    return _request("GET", "/model-info")


def samples() -> list[dict]:
    return _request("GET", "/samples")


def predict(flares: list[dict], save_run: bool, run_name: str | None) -> dict:
    return _request("POST", "/predict", json={"flares": flares, "save_run": save_run, "run_name": run_name})


def predict_file(filename: str, content: bytes, save_run: bool, run_name: str | None) -> dict:
    data = {"save_run": str(save_run).lower()}
    if run_name:
        data["run_name"] = run_name
    return _request("POST", "/predict/file", files={"file": (filename, content, "text/csv")}, data=data)


def runs() -> list[dict]:
    return _request("GET", "/runs")


def run(run_id: str) -> dict:
    return _request("GET", f"/runs/{run_id}")


def delete_run(run_id: str) -> None:
    _request("DELETE", f"/runs/{run_id}")
