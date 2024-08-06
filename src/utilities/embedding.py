import chromadb
from fastapi import HTTPException

from src.utilities.general import HOST, CHROMA_PORT

chroma_client = chromadb.HttpClient(
    host=HOST,
    port=CHROMA_PORT
)

basic_collection = chroma_client.get_or_create_collection('basic')


def add_record(title, content):
    try:
        basic_collection.add(
            ids=[title],
            documents=[content],
        )
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error adding record: {title}")


def update_record(title, content):
    try:
        basic_collection.update(
            ids=[title],
            documents=[content]
        )
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error updating record: {title}")


def delete_record(title):
    try:
        basic_collection.delete(
            ids=[title]
        )
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error deleting record: {title}")


def get_record(title):
    try:
        return basic_collection.get(
            ids=[title]
        )
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error getting record: {title}")


def query_record(query_embedding, max_results=5):
    try:
        return basic_collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
        )
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error querying record: {query_embedding}")