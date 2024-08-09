from typing import Union
from fastapi import APIRouter, Form, Header
from src.authentication.authentication import verify_token
from src.services.embedding import create_embedding
from src.utilities.general import NO_TOKEN

embedding_router = APIRouter(
    prefix="/Embedding",
    responses={
        200: {"description": "Successful"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
    tags=["Embedding Endpoints"]
)


@embedding_router.post("/generate_embedding")
async def generate_embedding(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        text: str = Form(description="Text to generate embedding for.")
):
    assert verify_token(token)
    embedding = await create_embedding(text)
    return str(embedding)
