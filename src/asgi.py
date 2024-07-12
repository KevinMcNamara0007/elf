from src.utilities.general import start_llama_cpp, stop_llama_cpp, HOST, UVICORN_PORT, handle_sigterm
start_llama_cpp()
from contextlib import asynccontextmanager
import signal
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException
from starlette.responses import RedirectResponse

from src.controllers import inference  #, images
from src.utilities.exception_handlers import request_validation_exception_handler, http_exception_handler, \
    unhandled_exception_handler
from log_management.middleware import log_request_middleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        stop_llama_cpp()


elf = FastAPI(
    title="Expert LLM Framework",
    summary="Delivers responses from available expert models.",
    version="1",
    swagger_ui_parameters={
        "syntaxHighlight.theme": "obsidian",
        "docExpansion": "none"
    },
    lifespan=lifespan,
)

# Include Routers
elf.include_router(inference.inference_router)
# app.include_router(images.images_router)

# CORS Fixes
elf.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

elf.middleware("http")(log_request_middleware)
elf.add_exception_handler(RequestValidationError, request_validation_exception_handler)
elf.add_exception_handler(HTTPException, http_exception_handler)
elf.add_exception_handler(Exception, unhandled_exception_handler)


# Redirect to doc page
@elf.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


signal.signal(signal.SIGTERM, handle_sigterm)

if __name__ == "__main__":
    uvicorn.run(elf, host=HOST, port=int(UVICORN_PORT))
