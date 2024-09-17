from fastapi import APIRouter, Body, status, UploadFile, File
from src.services.inference import vision_text_inference
from src.utilities.inference import doc_extractor

inference_router = APIRouter(
    prefix="/Inference",
    responses={
        200: {"description": "Successful"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
    tags=["Inference Endpoints"]
)


@inference_router.post("/ask_an_expert", status_code=status.HTTP_200_OK)
async def ask_an_expert(
        prompt: str = Body(...)
):
    return await vision_text_inference(prompt)


@inference_router.post("/process_document/")
async def process_document(
        file: UploadFile = File(...)
):
    # After extracting the text, you can pass it to your inference model
    return await doc_extractor(file)