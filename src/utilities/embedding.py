import chromadb
from chromadb.api.types import IncludeEnum
from chromadb.errors import DuplicateIDError
from fastapi import HTTPException

from src.utilities.general import HOST, CHROMA_PORT

chroma_client = chromadb.HttpClient(
    host=HOST,
    port=CHROMA_PORT
)

basic_collection = chroma_client.get_or_create_collection('basic')


def add_record(titles, contents, metadata=None):
    try:
        basic_collection.add(
            ids=titles,
            metadatas=metadata,
            documents=contents,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid Data: {str(e)}")
    except DuplicateIDError as e:
        raise HTTPException(status_code=400, detail=f"Duplicate ID: {str(e)}")
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error adding record: {titles}")


def update_record(titles, contents):
    try:
        basic_collection.update(
            ids=titles,
            documents=contents
        )
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error updating record: {titles}")


def delete_record(titles):
    try:
        basic_collection.delete(
            ids=titles
        )
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error deleting record: {titles}")


def get_record(titles, text_to_find=None, metadata=None, limit=None):
    try:
        return basic_collection.get(
            ids=titles,
            where=metadata,
            limit=limit,
            include=[IncludeEnum.documents, IncludeEnum.distances],
            where_document={'$contains': [text_to_find]} if text_to_find else None,
        )
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error getting record: {titles}")


def query_record(query_embedding, max_results=5):
    try:
        return basic_collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
        )
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error querying record: {query_embedding}")


def get_available_record_collections():
    try:
        return chroma_client.list_collections()
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error listing available record collections: {str(e)}")
