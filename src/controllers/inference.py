from fastapi import APIRouter, Body, status
from src.services.inference import vision_text_inference

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
    """
    Inference endpoint for asking an expert.
    :param prompt:
    :return:
    """
    return await vision_text_inference(prompt)
