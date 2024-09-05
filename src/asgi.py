from src.utilities.general import start_llama_cpp, start_chroma_db
print("Starting Llama.cpp")
start_llama_cpp()
print("Starting ChromaDB")
start_chroma_db()
print("Started aux servers...Starting FastAPI")
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException
from starlette.responses import RedirectResponse
from src.controllers import inference, crud
from src.utilities.exception_handlers import request_validation_exception_handler, http_exception_handler, \
    unhandled_exception_handler
from log_management.middleware import log_request_middleware, CacheRequestBodyMiddleware


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
elf.include_router(crud.crud_router)

# CORS Fixes
elf.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)
elf.add_middleware(CacheRequestBodyMiddleware)
elf.middleware("http")(log_request_middleware)
elf.add_exception_handler(RequestValidationError, request_validation_exception_handler)
elf.add_exception_handler(HTTPException, http_exception_handler)
elf.add_exception_handler(Exception, unhandled_exception_handler)


# Redirect to doc page
@elf.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")
