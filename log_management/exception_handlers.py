import json
import sys
from typing import Union
from fastapi import Request
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.exception_handlers import http_exception_handler as _http_exception_handler
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from log_management.logger import logger


async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Custom exception handler for request validation errors.
    :param request:
    :param exc:
    :return:
    """
    body = getattr(request.state, 'body', b'')  # Use cached body if available, default to empty bytes
    query_params = dict(request.query_params)  # Convert to dict for easier handling
    try:
        body_text = body.decode('utf-8')
    except UnicodeDecodeError:
        body_text = "<binary data>"

    detail = {"errors": exc.errors(), "body": body_text, "query_params": query_params}
    logger.info(detail)
    return JSONResponse(status_code=400, content=json.dumps(detail))


async def http_exception_handler(request: Request, exc: HTTPException) -> Union[JSONResponse, Response]:
    """
    This is a wrapper to the default HTTPException handler of FastAPI.
    This function will be called when a HTTPException is explicitly raised.
    :param request:
    :param exc:
    :return:
    """
    return await _http_exception_handler(request, exc)


async def unhandled_exception_handler(request: Request, exc: Exception) -> PlainTextResponse:
    """
    This middleware will log all unhandled exceptions.
    Unhandled Exceptions are all exceptions that are not HTTPExceptions or RequestValidationErrors.
    :param request:
    :param exc:
    :return:
    """
    host = getattr(getattr(request, "client", None), "host", None)
    port = getattr(getattr(request, "client", None), "port", None)
    url = f"{request.url.path}?{request.query_params}" if request.query_params else request.url.path
    exception_type, exception_value, exception_traceback = sys.exc_info()
    exception_name = getattr(exception_type, "__name__", None)
    logger.error(
        f'{host}:{port} - "{request.method} {url}" 500 Internal Server Error <{exception_name}: {exception_value}>'
    )
    return PlainTextResponse(str(exc), status_code=500)
