import chromadb
from chromadb.api.types import IncludeEnum
from chromadb.db.base import NotFoundError
from chromadb.errors import DuplicateIDError
from fastapi import HTTPException
from src.utilities.general import HOST, CHROMA_PORT

chroma_client = chromadb.HttpClient(
    host=HOST,
    port=CHROMA_PORT
)

def add_record(titles, contents, collection_name, metadata=None):
    try:
        collection = chroma_client.get_collection(collection_name)
        if not collection:
            raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")

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
        print(f"Add Record Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding record: {titles}")


def update_record(titles, contents, collection_name):
    try:
        collection = chroma_client.get_collection(collection_name)
        if not collection:
            raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")

        collection.update(
            ids=titles,
            documents=contents
        )
        return f"{collection_name} record updated"
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Record not found: {str(e)}")
    except Exception as e:
        print(f"Update Record Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating record: {titles}")


def delete_record(titles, collection_name):
    try:
        collection = chroma_client.get_collection(collection_name)
        if not collection:
            raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")

        collection.delete(
            ids=titles
        )
        return f"{collection_name} record deleted"
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Record not found: {str(e)}")
    except Exception as e:
        print(f"Delete Record Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting record: {titles}")


def get_record(titles, collection_name, text_to_find=None, metadata=None, limit=None):
    try:
        collection = chroma_client.get_collection(collection_name)
        if not collection:
            raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")

        return collection.get(
            ids=titles,
            where=metadata,
            limit=limit,
            include=[IncludeEnum.documents, IncludeEnum.distances],
            where_document={'$contains': [text_to_find]} if text_to_find else None,
        )
    except Exception as e:
        print(f"Get Record Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting record: {titles}")


def query_record(query_embedding, collection_name, max_results=5):
    try:
        collection = chroma_client.get_collection(collection_name)
        if not collection:
            raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")

        return collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
        )
    except Exception as e:
        print(f"Query Record Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying record: {query_embedding}")


def add_collection(collection_name):
    try:
        chroma_client.create_collection(
            collection_name
        )
        return f"{collection_name} collection created"
    except Exception as e:
        print(e)
        if "UniqueConstraintError" in str(e):
            raise HTTPException(status_code=400, detail=f"{collection_name} already exists.")
        raise HTTPException(status_code=500, detail=str(e))


def get_available_record_collections():
    try:
        coll_list = chroma_client.list_collections()
        groomed_collection_list = {}
        for collection in coll_list:
            name = collection.name
            metadata = collection.metadata if collection.metadata else {}
            groomed_collection_list.update({name: {"metadata": metadata,
                                                   "records": collection.count()}
                                            })
        return groomed_collection_list
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error listing available record collections: {str(e)}")


def remove_collection(collection_name):
    try:
        chroma_client.delete_collection(collection_name)
        return f"{collection_name} removed"
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error removing collection: {collection_name}")
