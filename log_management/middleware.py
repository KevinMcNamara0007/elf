import http
import time
from fastapi import Request
from log_management.logger import logger


async def log_request_middleware(request: Request, call_next):
    """
    This middleware will log all requests and their processing time
    E.g. log:
    0.0.0.0:123 - GET /ping 200 OK 1.00ms
    :param request:
    :param call_next:
    :return:
    """
    url = f"{request.url.path}?{request.query_params}" if request.query_params else request.url.path
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time)
    formatted_process_time = "{0:.4f}".format(process_time)
    host = getattr(getattr(request, "client", None), "host", None)
    port = getattr(getattr(request, "client", None), "port", None)
    try:
        status_phrase = http.HTTPStatus(response.status_code).phrase
    except ValueError:
        status_phrase = ""
    entry = f'{host}:{port} {url} {response.status_code} {status_phrase} {formatted_process_time}s'
    logger.info(entry)
    return response
