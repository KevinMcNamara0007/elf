from fastapi import APIRouter, Header, HTTPException, Body
from src.authentication.authentication import verify_token
from src.utilities.general import NO_TOKEN
from src.utilities.crud import (
    add_collection, add_record, update_record, delete_record,
    get_available_record_collections, remove_collection, get_record
)
from src.modeling.request_models import (
    GetRecordRequest, AddRecordRequest, UpdateRecordRequest, DeleteRecordRequest, AddCollectionRequest
)

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
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: GetRecordRequest = Body(...)
):
    """
    Returns record(s) based on title, text contained in the document.
    In progress feature: -> will return all records from a collection, offset is required.
    :param token:
    :param request:
    :return:
    """
    if not verify_token(token):
        raise HTTPException(status_code=403, detail="Invalid token")
    return get_record(
        titles=request.titles,  # required separated by :::
        collection_name=request.collection_name,  # required
        text_to_find=request.text_to_find,  # optional text to look for in documents
        metadata=request.metadata,  # optional not yet implemented
        limit=request.limit,  # optional number of records to return
    )


@crud_router.post("/add_record")
async def generate_record(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: AddRecordRequest = Body(...)
):
    """
    Adds new record(s) based on title, text contained in the document.
    In progress feature: -> metadata can be added to categorize documents further
    :param token:
    :param request:
    :return:
    """
    if not verify_token(token):
        raise HTTPException(status_code=403, detail="Invalid token")
    return add_record(
        titles=request.titles,  # required seperated by :::
        contents=request.contents,  # required seperated by :::
        collection_name=request.collection_name,  # required singular name
        metadata=request.metadata,  # optional not yet implemented
    )


@crud_router.post("/update_record")
async def modify_record(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: UpdateRecordRequest = Body(...)
):
    """
    Updates record(s) based on title, text contained in the document.
    In progress feature: -> update the metadata of a record
    :param token:
    :param request:
    :return:
    """
    if not verify_token(token):
        raise HTTPException(status_code=403, detail="Invalid token")
    return update_record(
        titles=request.titles,  # required seperated by :::
        contents=request.contents,  # required seperated by :::
        collection_name=request.collection_name,  # required singular name
    )


@crud_router.post("/delete_record")
async def remove_record(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: DeleteRecordRequest = Body(...)
):
    """
    Removes record(s) based on title, text contained in the document.
    :param token:
    :param request:
    :return:
    """
    if not verify_token(token):
        raise HTTPException(status_code=403, detail="Invalid token")
    return delete_record(
        titles=request.titles,  # required seperated by :::
        collection_name=request.collection_name,  # required singular name
    )


@crud_router.post("/get_collections")
async def get_collections(
        token: str = Header(default=NO_TOKEN, convert_underscores=False)
):
    """
    Returns all available collections.
    :param token:
    :return:
    """
    if not verify_token(token):
        raise HTTPException(status_code=403, detail="Invalid token")
    return get_available_record_collections()


@crud_router.post("/add_collection")
async def generate_collection(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: AddCollectionRequest = Body(...)
):
    """
    Adds new collection
    In progress feature: -> metadata can be added to a collection
    :param token:
    :param request:
    :return:
    """
    if not verify_token(token):
        raise HTTPException(status_code=403, detail="Invalid token")
    return add_collection(
        request.collection_name  # required singular name
    )


@crud_router.post("/delete_collection")
async def delete_collection(
        token: str = Header(default=NO_TOKEN, convert_underscores=False),
        request: AddCollectionRequest = Body(...)
):
    """
    Removes collection
    :param token:
    :param request:
    :return:
    """
    if not verify_token(token):
        raise HTTPException(status_code=403, detail="Invalid token")
    return remove_collection(
        request.collection_name  # required singular name
    )
