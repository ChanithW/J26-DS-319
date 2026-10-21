"""HTTP service around the Component 2 model (run: uvicorn component2.api:app)."""
from __future__ import annotations

from .main import create_app

__all__ = ["app", "create_app"]


def __getattr__(name: str):
    """Build ``app`` lazily so importing the package has no side effects (no DB file created)."""
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
