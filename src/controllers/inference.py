import json
from typing import Union
from fastapi import APIRouter, status, Form, HTTPException, Header
from src.authentication.authentication import verify_token
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
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        messages: str = Form(default=None, description="Chat style prompting"),
        prompt: str = Form(default=None, description="The prompt you want answered."),
        temperature: float = Form(default=0.05, description="Temperature of the model."),
        rules: str = Form(default="You are a friendly virtual assistant.",
                          description="Rules of the model."),
        top_k: int = Form(default=40),
        top_p: float = Form(default=0.95)
):
    assert verify_token(token)
    history = []
    # Validate and parse the messages
    if messages:
        try:
            history.extend(json.loads(messages))
            if not isinstance(history, list):
                raise HTTPException(status_code=400, detail="Messages must be a list of dictionaries.")
            for message in history:
                if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                    raise HTTPException(status_code=400,
                                        detail="Each message must be a dictionary with 'role' and 'content' keys.")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format for messages. {exc}")

    if prompt:
        history.append({"role": "User", "content": prompt})
    if not messages and not prompt:
        raise HTTPException(status_code=400, detail="Provide Messages, Prompt or both.")

    return await get_expert_response(
        rules=rules,
        messages=history,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


@inference_router.post("/classify", status_code=status.HTTP_200_OK, description="Classify your prompt.")
async def determine_expert(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        prompt: str = Form()
):
    assert verify_token(token)
    return await prompt_classification(prompt)


@inference_router.post("/semantic_search", status_code=status.HTTP_200_OK, description="Semantic search.")
async def semantic_search(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        query: str = Form(description="What you're looking for."),
        collection_name: str = Form(description="Collection name"),
        max_results: int = Form(default=5, description="Maximum number of results to return"),
):
    assert verify_token(token)
    return query_record(query, collection_name, max_results)
