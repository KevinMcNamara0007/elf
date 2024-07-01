import datetime
start = datetime.datetime.now()
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException
from starlette.responses import RedirectResponse
from log_management.middleware import log_request_middleware
from src.controllers import inference#, images
from src.utilities.exception_handlers import request_validation_exception_handler, http_exception_handler, \
    unhandled_exception_handler

elf = FastAPI(
    title="Expert LLM Framework",
    summary="Delivers responses from available expert models.",
    version="1",
    swagger_ui_parameters={
        "syntaxHighlight.theme": "obsidian",
        "docExpansion": "none"
    }
)
# Include Routers
elf.include_router(inference.inference_router)
# elf.include_router(images.images_router)

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

print(f"Application Loaded in: {(datetime.datetime.now() - start).total_seconds()}s")

# Redirect to doc page
@elf.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")
