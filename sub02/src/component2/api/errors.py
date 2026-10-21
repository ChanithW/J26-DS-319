"""One JSON error shape: {"error": {"code", "message", "details"}}. No stack traces."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)


class ApiError(Exception):
    """Raise to return a structured error response."""

    def __init__(self, status: int, code: str, message: str, details=None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def _resp(status: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(status_code=status,
                        content={"error": {"code": code, "message": message, "details": details}})


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api(_: Request, e: ApiError):
        return _resp(e.status, e.code, e.message, e.details)

    @app.exception_handler(RequestValidationError)
    async def _val(_: Request, e: RequestValidationError):
        details = [{"loc": [str(x) for x in err["loc"]], "message": err["msg"]} for err in e.errors()]
        return _resp(422, "validation_error", "The request body is not valid.", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, e: StarletteHTTPException):
        return _resp(e.status_code, "http_error", str(e.detail))

    @app.exception_handler(Exception)
    async def _any(_: Request, e: Exception):
        log.exception("unhandled error")
        return _resp(500, "internal_error", "Something went wrong on the server.")
