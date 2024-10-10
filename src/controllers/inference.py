from fastapi import APIRouter, HTTPException, Header, Body
from starlette import status
from starlette.responses import StreamingResponse
from src.authentication.authentication import verify_token
from src.modeling.request_models import AskExpertRequest, ClassifyRequest, SemanticSearchRequest, Message, Pro
from src.services.inference import get_expert_response, prompt_classification, get_expert_response_stream, \
    get_pro_response, get_pro_response_stream
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
    """
    Ask any question to one of the LLMs.
    :param token:
    :param request:
    :return:
    """
    assert verify_token(token)
    history = request.messages or []

    if request.prompt:
        message = Message()
        message.role = "User"
        message.content = request.prompt
        history.append(message)
    if not history:
        raise HTTPException(status_code=400, detail="Provide Messages, Prompt or both.")

    return await get_expert_response(
        rules=request.rules,  # optional the role the LLM should play.
        messages=history,  # optional must be formatted "[{"role": "system", "content": "system prompt"}]"
        temperature=request.temperature,  # optional temperature of the LLM
        top_k=request.top_k,  # optional Number of words to consider for next token
        top_p=request.top_p,  # optional Percentage to limit next token generation to
    )


@inference_router.post("/ask_an_expert_stream", status_code=status.HTTP_200_OK, description="Ask any question.")
async def ask_an_expert_stream(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: AskExpertRequest = Body(...)
):
    assert verify_token(token)
    history = request.messages or []

    if request.prompt:
        message = Message()
        message.role = "User"
        message.content = request.prompt
        history.append(message)
    if not history:
        raise HTTPException(status_code=400, detail="Provide Messages, Prompt or both.")

    # Stream the result directly from the get_expert_response_stream function
    return StreamingResponse(
        get_expert_response_stream(
            rules=request.rules,  # optional the role the LLM should play.
            messages=history,  # optional must be formatted "[{"role": "system", "content": "system prompt"}]"
            temperature=request.temperature,  # optional temperature of the LLM
            top_k=request.top_k,  # optional Number of words to consider for next token
            top_p=request.top_p,  # optional Percentage to limit next token generation to
        ),
        media_type="text/plain"
    )


@inference_router.post("/ask_a_pro_stream", status_code=status.HTTP_200_OK, description="Digital Professional Instructions")
async def ask_a_pro_stream(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: Pro = Body(...)
):
    """
    Ask any question to one of the LLMs.
    :param token:
    :param request:
    :return:
    """
    assert verify_token(token)

    if request.prompt:
        return StreamingResponse(
            get_pro_response_stream(
                prompt=request.prompt,  # optional the role the LLM should play.
                output_tokens=request.output_tokens
            ),
            media_type="text_plain"
        )
    else:
        raise HTTPException(status_code=400, detail="Provide a prompt")


@inference_router.post("/ask_a_pro", status_code=status.HTTP_200_OK, description="Digital Professional Instructions")
async def ask_a_pro(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: Pro = Body(...)
):
    """
    Ask any question to one of the LLMs.
    :param token:
    :param request:
    :return:
    """
    assert verify_token(token)

    if request.prompt:
        return await get_pro_response(
            prompt=request.prompt,  # optional the role the LLM should play.
        )
    else:
        raise HTTPException(status_code=400, detail="Provide a prompt")


@inference_router.post("/classify", status_code=status.HTTP_200_OK, description="Classify your prompt.")
async def determine_expert(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: ClassifyRequest = Body(...)
):
    """
    Classify your prompt.
    :param token:
    :param request:
    :return:
    """
    assert verify_token(token)
    return await prompt_classification(
        request.prompt  # required prompt to classify
    )


@inference_router.post("/semantic_search", status_code=status.HTTP_200_OK, description="Semantic search.")
async def semantic_search(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: SemanticSearchRequest = Body(...)
):
    assert verify_token(token)
    return query_record(
        request.query,  # required prompt to query against
        request.collection_name,   # required collection to retrieve records from
        request.max_results  # optional max number of results to return
    )
