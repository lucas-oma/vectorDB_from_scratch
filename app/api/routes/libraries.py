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
    lib = await svc.create_library(body.name, body.dims, body.index_type or "flat", body.metadata)
    return LibraryResponse(**lib.dict())

@router.get("/", response_model=List[LibraryResponse])
async def list_libraries(svc: VectorDBService = Depends(get_service)):
    libs = await svc.list_libraries()
    libs.sort(key=lambda x: (x.name.lower(), x.id))
    return [LibraryResponse(**l.dict()) for l in libs]

@router.get("/{library_id}", response_model=LibraryResponse)
async def get_library(library_id: str, svc: VectorDBService = Depends(get_service)):
    lib = await svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library not found")
    return LibraryResponse(**lib.dict())

@router.patch("/{library_id}", response_model=LibraryResponse)
async def update_library(library_id: str, body: UpdateLibraryRequest, svc: VectorDBService = Depends(get_service)):
    lib = await svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library not found")
    updates = {}
    if body.name is not None: updates["name"] = body.name
    if body.index_type is not None: updates["index_type"] = body.index_type
    if body.metadata is not None: updates["metadata"] = body.metadata
    
    # if no updates, return original library
    if updates:
        updated = await svc.update_library(library_id, updates)
        if updated:
            return LibraryResponse(**updated.dict())
    
    # No updates or update returned None (no changes needed)
    return LibraryResponse(**lib.dict())

@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library(library_id: str, svc: VectorDBService = Depends(get_service)):
    ok = await svc.delete_library(library_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")
    return None
