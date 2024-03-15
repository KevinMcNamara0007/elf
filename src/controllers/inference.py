from fastapi import APIRouter, status, Form

from src.services.inference import get_all_models, get_expert_response

inference_router = APIRouter(
    prefix="/Inference",
    responses={
        200: {"description": "Successful"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
    tags=["Inference Endpoints"]
)


@inference_router.get("/all_models",
                      status_code=status.HTTP_200_OK,
                      description="Returns a list of the available models."
                      )
async def fetch_all_models():
    return await get_all_models()


@inference_router.post("/ask_an_expert", status_code=status.HTTP_200_OK, description="Ask any question.")
async def ask_an_expert(
        prompt: str = Form(description="Prompt you seek an answer to."),
        temperature: float = Form(default=.05, description="Temperature of the model.")
):
    return await get_expert_response(prompt=prompt, temperature=temperature)
