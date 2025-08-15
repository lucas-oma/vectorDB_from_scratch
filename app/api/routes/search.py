from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_service
from app.core.vector_db import VectorDBService
from app.api.dto import (
    RebuildIndexRequest, SearchRequest,
    SearchResponse, LibraryStatsResponse, ChunkResponse
)

router = APIRouter()

@router.post("/{library_id}/index/rebuild", status_code=status.HTTP_202_ACCEPTED)
async def rebuild_index(library_id: str, body: RebuildIndexRequest, svc: VectorDBService = Depends(get_service)):
    lib = svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")
    # MVP: only 'linear' supported
    idx_type = (body.index_type or lib.index_type or "linear").lower()
    if idx_type != "linear":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Only 'linear' index supported in MVP (got '{idx_type}')")
    svc.rebuild_index(library_id)
    return {"message": f"Index rebuilt for library '{library_id}'", "index_type": "linear", "dims": lib.dims}

@router.get("/{library_id}/stats", response_model=LibraryStatsResponse)
async def get_library_stats(library_id: str, svc: VectorDBService = Depends(get_service)):
    lib = svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")
    docs = svc.list_documents(library_id)
    chs = svc.list_chunks(library_id)
    idx_built = library_id in svc.indexes
    return LibraryStatsResponse(
        library_id=library_id,
        name=lib.name,
        dims=lib.dims,
        index_type=lib.index_type,
        num_documents=len(docs),
        num_chunks=len(chs),
        index_built=bool(idx_built),
    )

@router.post("/{library_id}/search", response_model=SearchResponse)
async def search(library_id: str, body: SearchRequest, svc: VectorDBService = Depends(get_service)):
    try:
        results = svc.search(library_id, body.embedding, body.k, include_chunk=body.include_chunk)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    out = []
    for r in results:
        chunk_resp = ChunkResponse(**r.chunk.dict()) if (body.include_chunk and r.chunk) else None
        out.append({"chunk_id": r.chunk_id, "similarity_score": r.similarity_score, "chunk": chunk_resp})

    return SearchResponse(library_id=library_id, results=out)
