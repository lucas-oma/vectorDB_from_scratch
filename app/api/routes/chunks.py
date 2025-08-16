from __future__ import annotations
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_service
from app.core.vector_db import VectorDBService
from app.api.dto import (
    CreateChunkRequest, CreateChunksBatchRequest, UpdateChunkRequest,
    ChunkResponse, CreateChunksResponse, DeleteChunksBatchRequest
)

router = APIRouter()

@router.post("/{library_id}/chunks", response_model=ChunkResponse, status_code=status.HTTP_201_CREATED)
async def create_chunk(library_id: str, body: CreateChunkRequest, svc: VectorDBService = Depends(get_service)):
    try:
        ch = await svc.create_chunk(library_id, body.document_id, body.text, body.embedding, body.metadata)
        return ChunkResponse(**ch.dict())
    except KeyError as e:  # "library" or "document"
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:  # dim mismatch
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

@router.post("/{library_id}/chunks/batch", response_model=CreateChunksResponse, status_code=status.HTTP_201_CREATED)
async def create_chunks_batch(library_id: str, body: CreateChunksBatchRequest, svc: VectorDBService = Depends(get_service)):
    ids: List[str] = []
    try:
        for item in body.chunks:
            ch = await svc.create_chunk(library_id, item.document_id, item.text, item.embedding, item.metadata)
            ids.append(ch.id)
        return CreateChunksResponse(chunk_ids=ids)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

@router.get("/{library_id}/chunks", response_model=List[ChunkResponse])
async def list_chunks(library_id: str, svc: VectorDBService = Depends(get_service)):
    try:
        items = await svc.list_chunks(library_id)  # simple wrapper over storage.load_chunks_for_library
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")
    return [ChunkResponse(**c.dict()) for c in items]

@router.get("/{library_id}/chunks/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(library_id: str, chunk_id: str, svc: VectorDBService = Depends(get_service)):
    lib = await svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")
    ch = await svc.get_chunk(chunk_id)
    if not ch or ch.library_id != library_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="chunk")
    return ChunkResponse(**ch.dict())

@router.patch("/{library_id}/chunks/{chunk_id}", response_model=ChunkResponse)
async def update_chunk(library_id: str, chunk_id: str, body: UpdateChunkRequest, svc: VectorDBService = Depends(get_service)):
    try:
        # First verify the chunk exists
        ch = await svc.get_chunk(chunk_id)
        if not ch or ch.library_id != library_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="chunk")
        
        updates = {}
        if body.text is not None: updates["text"] = body.text
        if body.embedding is not None: updates["embedding"] = body.embedding
        if body.metadata is not None: updates["metadata"] = body.metadata
        
        # if no updates, return original chunk
        if updates:
            updated = await svc.update_chunk(library_id, chunk_id, **updates)
            if updated:
                return ChunkResponse(**updated.dict())
        
        # No updates or update returned None (no changes needed)
        return ChunkResponse(**ch.dict())
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

@router.delete("/{library_id}/chunks/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chunk(library_id: str, chunk_id: str, svc: VectorDBService = Depends(get_service)):
    ok = await svc.delete_chunk(library_id, chunk_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="chunk")
    return None

@router.delete("/{library_id}/chunks", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chunks_batch(library_id: str, body: DeleteChunksBatchRequest, svc: VectorDBService = Depends(get_service)):
    # validate all exist first for simple atomic semantics
    for cid in body.chunk_ids:
        ch = await svc.get_chunk(cid)
        if not ch or ch.library_id != library_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"chunk '{cid}'")
    for cid in body.chunk_ids:
        await svc.delete_chunk(library_id, cid)
    return None
