from typing import Union, Optional
from fastapi import APIRouter, Form, Header
from pydantic import BaseModel, Field
from src.authentication.authentication import verify_token
from src.utilities.general import NO_TOKEN
from src.utilities.crud import add_collection, add_record, update_record, delete_record, \
    get_available_record_collections, remove_collection

crud_router = APIRouter(
    prefix="/CRUD",
    responses={
        200: {"description": "Successful"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
    tags=["CRUD Endpoints"]
)


# Model to encapsulate the data being passed
class RecordData(BaseModel):
    titles: list[str] = Field(description="Titles of record")
    contents: list[str] = Field(description="Contents of record")
    collection_name: str = Field(description="Name of collection")
    metadata: Optional[dict] = Field(default=None)


@crud_router.post("/add_record")
async def generate_record(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        data: RecordData = Form(...)
):
    assert verify_token(token)
    return add_record(
        titles=data.titles, contents=data.contents, collection_name=data.collection_name, metadata=data.metadata
    )


@crud_router.post("/update_record")
async def modify_record(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        data: RecordData = Form(...)
):
    assert verify_token(token)
    return update_record(
        titles=data.titles, contents=data.contents, collection_name=data.collection_name
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


@crud_router.post("/get_collections")
async def get_collections(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False)
):
    assert verify_token(token)
    return get_available_record_collections()


@crud_router.post("/add_collection")
async def generate_collection(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        collection_name: str = Form(description="Name of collection")
):
    assert verify_token(token)
    return add_collection(collection_name)


@crud_router.post("/delete_collection")
async def delete_collection(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        collection_name: str = Form(description="Name of collection"),
):
    assert verify_token(token)
    return remove_collection(collection_name)
