from __future__ import annotations
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_service
from app.core.vector_db import VectorDBService
from app.api.dto import (
    CreateLibraryRequest, UpdateLibraryRequest,
    LibraryResponse
)

router = APIRouter()

@router.post("/", response_model=LibraryResponse, status_code=status.HTTP_201_CREATED)
async def create_library(body: CreateLibraryRequest, svc: VectorDBService = Depends(get_service)):
    lib = svc.create_library(body.name, body.dims, body.index_type or "linear", body.metadata)
    return LibraryResponse(**lib.dict())

@router.get("/", response_model=List[LibraryResponse])
async def list_libraries(svc: VectorDBService = Depends(get_service)):
    libs = svc.list_libraries()
    # deterministic order by name then id
    libs.sort(key=lambda x: (x.name.lower(), x.id))
    return [LibraryResponse(**l.dict()) for l in libs]

@router.get("/{library_id}", response_model=LibraryResponse)
async def get_library(library_id: str, svc: VectorDBService = Depends(get_service)):
    lib = svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")
    return LibraryResponse(**lib.dict())

@router.patch("/{library_id}", response_model=LibraryResponse)
async def update_library(library_id: str, body: UpdateLibraryRequest, svc: VectorDBService = Depends(get_service)):
    lib = svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")
    updates = {}
    if body.name is not None: updates["name"] = body.name
    if body.index_type is not None: updates["index_type"] = body.index_type
    if body.metadata is not None: updates["metadata"] = body.metadata
    updated = svc.update_library(library_id, updates)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")
    return LibraryResponse(**updated.dict())

@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(library_id: str, svc: VectorDBService = Depends(get_service)):
    ok = svc.delete_library(library_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")
    return None
