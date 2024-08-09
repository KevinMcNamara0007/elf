from typing import Union
from fastapi import APIRouter, Form, Header
from src.authentication.authentication import verify_token
from src.utilities.general import NO_TOKEN
from src.utilities.crud import add_collection, add_record, update_record, delete_record

crud_router = APIRouter(
    prefix="/CRUD",
    responses={
        200: {"description": "Successful"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
    tags=["CRUD Endpoints"]
)


@crud_router.post("/add_collection")
async def generate_collection(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        collection_name: str = Form(description="Name of collection")
):
    assert verify_token(token)
    return add_collection(collection_name)


@crud_router.post("/add_record")
async def generate_record(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        titles: list[str] = Form(description="Titles of record"),
        contents: list[str] = Form(description="Contents of record"),
        collection_name: str = Form(description="Name of collection"),
        metadata: dict = Form(default=None)
):
    assert verify_token(token)
    return add_record(
        titles=titles, contents=contents, collection_name=collection_name, metadata=metadata
    )


@crud_router.post("/update_record")
async def modify_record(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        titles: list[str] = Form(description="Titles of record"),
        contents: list[str] = Form(description="Contents of record"),
        collection_name: str = Form(description="Name of collection"),
):
    assert verify_token(token)
    return update_record(
        titles=titles, contents=contents, collection_name=collection_name
    )


@crud_router.post("/delete_record")
async def remove_record(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        titles: list[str] = Form(description="Titles of record"),
        collection_name: str = Form(description="Name of collection"),
):
    assert verify_token(token)
    return delete_record(
        titles=titles, collection_name=collection_name
    )

