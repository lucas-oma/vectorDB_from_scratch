from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


# Request models
class CreateLibraryRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Library name")
    dims: int = Field(..., gt=0, description="Embedding dimensions")
    index_type: Optional[str] = Field(default="flat", description="Index type")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="Library metadata")


class UpdateLibraryRequest(BaseModel):
    name: Optional[str] = None
    index_type: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


class CreateDocumentRequest(BaseModel):
    title: str = Field(..., description="Document title")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="Document metadata")


class UpdateDocumentRequest(BaseModel):
    title: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


class CreateChunkRequest(BaseModel):
    document_id: str = Field(..., description="Parent document ID")
    text: str = Field(..., description="Chunk text content")
    embedding: List[float] = Field(..., description="Chunk embedding vector")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="Chunk metadata")


class CreateChunksBatchRequest(BaseModel):
    chunks: List[CreateChunkRequest] = Field(..., description="List of chunks to create")


class UpdateChunkRequest(BaseModel):
    text: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, str]] = None


class RebuildIndexRequest(BaseModel):
    index_type: Optional[str] = Field(default="linear", description="Index type")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Index parameters")


class SearchRequest(BaseModel):
    embedding: List[float] = Field(..., description="Query embedding vector")
    k: int = Field(default=10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    include_chunk: bool = Field(default=False, description="Include chunk data in results")


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="Texts to embed")
    model: Optional[str] = Field(default="embed-english-v3.0", description="Embedding model")


class DeleteChunksBatchRequest(BaseModel):
    chunk_ids: List[str] = Field(..., description="List of chunk IDs to delete")


# Response models
class LibraryResponse(BaseModel):
    id: str
    name: str
    dims: int
    index_type: str
    metadata: Optional[Dict[str, str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DocumentResponse(BaseModel):
    id: str
    library_id: str
    title: str
    metadata: Optional[Dict[str, str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ChunkResponse(BaseModel):
    id: str
    library_id: str
    document_id: str
    text: str
    embedding: List[float]
    metadata: Optional[Dict[str, str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class LibraryStatsResponse(BaseModel):
    library_id: str
    num_documents: int
    num_chunks: int
    dims: int
    index_type: str


class SearchResultResponse(BaseModel):
    chunk_id: str
    similarity_score: float
    chunk: Optional[ChunkResponse] = None


class SearchResponse(BaseModel):
    library_id: str
    results: List[SearchResultResponse]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


class CreateChunksResponse(BaseModel):
    chunk_ids: List[str]


class SnapshotMetadataResponse(BaseModel):
    id: str
    library_id: str
    created_at: datetime
    num_documents: int
    num_chunks: int
    index_type: str


class VersionResponse(BaseModel):
    version: str = "0.1.0"

