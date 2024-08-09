import chromadb
from chromadb.api.types import IncludeEnum
from chromadb.errors import DuplicateIDError
from fastapi import HTTPException
from src.services.embedding import create_embedding
from src.utilities.general import HOST, CHROMA_PORT

chroma_client = chromadb.HttpClient(
    host=HOST,
    port=CHROMA_PORT
)


def add_collection(collection_name):
    try:
        chroma_client.create_collection(
            collection_name,
            embedding_function=create_embedding
        )
        return f"{collection_name} collection created"
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


def add_record(titles, contents, collection_name, metadata=None):
    try:
        collection = chroma_client.get_collection(collection_name)
        collection.add(
            ids=titles,
            metadatas=metadata,
            documents=contents,
        )
        return f"{collection_name} record created"
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid Data: {str(e)}")
    except DuplicateIDError as e:
        raise HTTPException(status_code=400, detail=f"Duplicate ID: {str(e)}")
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error adding record: {titles}")


def update_record(titles, contents, collection_name):
    try:
        collection = chroma_client.get_collection(collection_name)
        collection.update(
            ids=titles,
            documents=contents
        )
        return f"{collection_name} record updated"
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error updating record: {titles}")


def delete_record(titles, collection_name):
    try:
        collection = chroma_client.get_collection(collection_name)
        collection.delete(
            ids=titles
        )
        return f"{collection_name} record deleted"
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error deleting record: {titles}")


def get_record(titles, collection_name, text_to_find=None, metadata=None, limit=None):
    try:
        collection = chroma_client.get_collection(collection_name)
        return collection.get(
            ids=titles,
            where=metadata,
            limit=limit,
            include=[IncludeEnum.documents, IncludeEnum.distances],
            where_document={'$contains': [text_to_find]} if text_to_find else None,
        )
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error getting record: {titles}")


def query_record(query_embedding, collection_name, max_results=5):
    try:
        collection = chroma_client.get_collection(collection_name)
        return collection.query(
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
