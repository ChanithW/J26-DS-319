"""FastAPI application factory. The bundle is loaded once at start-up."""
from __future__ import annotations

import contextvars
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from ..bundle import load_bundle
from .db import make_session_factory
from .errors import install_handlers
from .routes import router
from .settings import Settings

log = logging.getLogger("component2.api")
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Pass ``settings`` in tests; otherwise environment variables are read."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle, app.state.bundle_error = None, None
        try:
            app.state.bundle = load_bundle(settings.component2_bundle_path)
            log.info("model bundle loaded (version %s)", app.state.bundle["version"])
        except Exception as e:                           # /health then answers 503
            app.state.bundle_error = str(e)
            log.error("model bundle not loaded: %s", e)
        yield

    app = FastAPI(title="Component 2: CME occurrence prediction", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.session_factory = make_session_factory(settings.database_url)
    app.state.bundle, app.state.bundle_error = None, None
    install_handlers(app)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list, allow_methods=["*"],
                       allow_headers=["*"], expose_headers=["Content-Disposition", "X-Request-ID"])

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request_id_var.set(rid)
        log.info("%s %s", request.method, request.url.path)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    for h in logging.getLogger().handlers:
        h.addFilter(_RequestIdFilter())
    app.include_router(router)
    return app
