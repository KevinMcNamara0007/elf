from typing import List, Optional, Any
from pydantic import BaseModel


class GetRecordRequest(BaseModel):
    titles: str
    collection_name: str
    text_to_find: Optional[str] = None
    metadata: Optional[str] = None
    limit: Optional[int] = None


class AddRecordRequest(BaseModel):
    titles: str
    contents: str
    collection_name: str
    metadata: Optional[str] = None


class UpdateRecordRequest(BaseModel):
    titles: str
    contents: str
    collection_name: str


class DeleteRecordRequest(BaseModel):
    titles: List[str]
    collection_name: str


class AddCollectionRequest(BaseModel):
    collection_name: str


class Message(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class AskExpertRequest(BaseModel):
    temperature: float = 0.05
    rules: str = "You are a virtual assistant."
    top_k: int = 40
    top_p: float = .95
    messages: Optional[List[Message]] = None
    prompt: Optional[str] = None


class ClassifyRequest(BaseModel):
    prompt: str


class SemanticSearchRequest(BaseModel):
    query: str
    collection_name: str
    max_results: int = 5
