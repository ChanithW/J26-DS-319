"""Write docs/openapi.json (run: python -m component2.api.dump_openapi)."""
from __future__ import annotations

import json
from pathlib import Path

from .main import create_app
from .settings import Settings

OPENAPI_PATH = Path(__file__).resolve().parents[3] / "docs" / "openapi.json"


def build_schema() -> dict:
    return create_app(Settings(database_url="sqlite://")).openapi()


if __name__ == "__main__":
    OPENAPI_PATH.parent.mkdir(exist_ok=True)
    OPENAPI_PATH.write_text(json.dumps(build_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OPENAPI_PATH}")
