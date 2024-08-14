from fastapi import APIRouter, HTTPException, Header, Body
from starlette import status
from src.authentication.authentication import verify_token
from src.modeling.request_models import AskExpertRequest, ClassifyRequest, SemanticSearchRequest, Message
from src.services.inference import get_expert_response, prompt_classification
from src.utilities.crud import query_record
from src.utilities.general import NO_TOKEN

inference_router = APIRouter(
    prefix="/Inference",
    responses={
        200: {"description": "Successful"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
    tags=["Inference Endpoints"]
)


@inference_router.post("/ask_an_expert", status_code=status.HTTP_200_OK, description="Ask any question.")
async def ask_an_expert(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: AskExpertRequest = Body(...)
):
    assert verify_token(token)
    history = request.messages or []

    if request.prompt:
        message = Message()
        message.role = "user"
        message.content = request.prompt
        history.append(message)
    if not history:
        raise HTTPException(status_code=400, detail="Provide Messages, Prompt or both.")

    return await get_expert_response(
        rules=request.rules,
        messages=history,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )


@inference_router.post("/classify", status_code=status.HTTP_200_OK, description="Classify your prompt.")
async def determine_expert(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: ClassifyRequest = Body(...)
):
    assert verify_token(token)
    return await prompt_classification(request.prompt)


@inference_router.post("/semantic_search", status_code=status.HTTP_200_OK, description="Semantic search.")
async def semantic_search(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: SemanticSearchRequest = Body(...)
):
    assert verify_token(token)
    return query_record(request.query, request.collection_name, request.max_results)
