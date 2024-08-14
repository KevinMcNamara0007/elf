from typing import Union
from fastapi import APIRouter, Form, Header
from src.authentication.authentication import verify_token
from src.utilities.general import NO_TOKEN
from src.utilities.crud import add_collection, add_record, update_record, delete_record, \
    get_available_record_collections, remove_collection, get_record

crud_router = APIRouter(
    prefix="/CRUD",
    responses={
        200: {"description": "Successful"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
    tags=["CRUD Endpoints"]
)


@crud_router.post("/get_record")
async def fetch_record(
        token: Union[str, None] = Header(NO_TOKEN, convert_underscores=False),
        titles: str = Form(default=None, description="List of titles to fetch."),
        collection_name: str = Form(description="Name of the collection to fetch."),
        text_to_find: str = Form(default=None, description="Text to find to find."),
        metadata: str = Form(default=None, description="Metadata to find."),
        limit: int = Form(default=None, description="Number of records to fetch."),
):
    assert verify_token(token)
    return get_record(
        titles=titles,
        collection_name=collection_name,
        text_to_find=text_to_find,
        metadata=metadata,
        limit=limit,
    )


@crud_router.post("/add_record")
async def generate_record(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        titles: str = Form(description="List of titles to add."),
        contents: str = Form(description="List of contents to add."),
        metadata: str = Form(default=None, description="Dictionary of metadata to add."),
        collection_name: str = Form(description="Name of the collection to add the record to."),
):
    assert verify_token(token)
    return add_record(
        titles=titles, contents=contents, collection_name=collection_name, metadata=metadata
    )


@crud_router.post("/update_record")
async def modify_record(
        token: Union[str, None] = Header(default=NO_TOKEN, convert_underscores=False),
        titles: str = Form(description="List of titles to add."),
        contents: str = Form(description="List of contents to add."),
        collection_name: str = Form(description="Name of the collection to add the record to."),
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
