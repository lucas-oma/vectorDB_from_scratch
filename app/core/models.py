import uuid
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
# from datetime import datetime

"""
Data models
"""

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the chunk")
    document_id: str = Field(..., description="ID of the parent document")
    library_id: str = Field(..., description="ID of the parent library")
    text: str = Field(..., min_length=1, description="The text content of the chunk")
    embedding: List[float] = Field(..., min_items=1, description="Vector embedding of the chunk")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata")
    # created_at: datetime = Field(default_factory=datetime.utcnow) # TODO: check if needed or if just metadata is enough
    # updated_at: datetime = Field(default_factory=datetime.utcnow) # TODO: check if needed or if just metadata is enough

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the document")
    library_id: str = Field(..., description="ID of the parent library")
    title: str = Field(..., min_length=1, description="Title of the document")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Document-level metadata")
    # created_at: datetime = Field(default_factory=datetime.utcnow) # TODO: check if needed or if just metadata is enough
    # updated_at: datetime = Field(default_factory=datetime.utcnow) # TODO: check if needed or if just metadata is enough

class Library(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the library")
    name: str = Field(..., min_length=1, description="Name of the library")
    dims: int = Field(..., gt=0, description="Embedding dimensionality for this library's content")
    index_type: str = Field(default="linear", description="Indexing algorithm: linear, kdtree, or lsh")  # "linear" | "kdtree" | "lsh"
    metadata: Dict[str, str] = Field(default_factory=dict, description="Library-level metadata")
    # created_at: datetime = Field(default_factory=datetime.utcnow) # TODO: check if needed or if just metadata is enough
    # updated_at: datetime = Field(default_factory=datetime.utcnow) # TODO: check if needed or if just metadata is enough



"""
Search-related models
"""

class SearchResult(BaseModel):
    chunk_id: str = Field(..., description="ID of the matching chunk")
    similarity_score: float = Field(..., description="Similarity score (can be negative for distances)")
    chunk: Optional[Chunk] = Field(None, description="The matching chunk object (if available)")

class SearchQuery(BaseModel):
    embedding: List[float] = Field(..., min_items=1, description="Query embedding vector")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    metadata_filters: Optional[Dict[str, str]] = Field(None, description="Metadata filters to apply")
