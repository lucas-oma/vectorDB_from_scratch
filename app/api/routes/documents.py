from __future__ import annotations
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_service
from app.core.vector_db import VectorDBService
from app.api.dto import (
    CreateDocumentRequest, UpdateDocumentRequest,
    DocumentResponse
)

router = APIRouter()

@router.post("/{library_id}/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(library_id: str, body: CreateDocumentRequest, svc: VectorDBService = Depends(get_service)):
    try:
        doc = svc.create_document(library_id, body.title, body.metadata)
        return DocumentResponse(**doc.dict())
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")

@router.get("/{library_id}/documents", response_model=List[DocumentResponse])
async def list_documents(library_id: str, svc: VectorDBService = Depends(get_service)):
    lib = svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")
    docs = svc.list_documents(library_id)
    return [DocumentResponse(**d.dict()) for d in docs]

@router.get("/{library_id}/documents/{document_id}", response_model=DocumentResponse)
async def get_document(library_id: str, document_id: str, svc: VectorDBService = Depends(get_service)):
    lib = svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="library")
    doc = svc.get_document(document_id)
    if not doc or doc.library_id != library_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document")
    return DocumentResponse(**doc.dict())

@router.patch("/{library_id}/documents/{document_id}", response_model=DocumentResponse)
async def update_document(library_id: str, document_id: str, body: UpdateDocumentRequest, svc: VectorDBService = Depends(get_service)):
    try:
        updates = {}
        if body.title is not None: updates["title"] = body.title
        if body.metadata is not None: updates["metadata"] = body.metadata
        # library_id changes are not allowed by the VectorDb service; aka. don't pass it
        updated = svc.update_document(document_id, updates)
        if not updated or updated.library_id != library_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document")
        return DocumentResponse(**updated.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

@router.delete("/{library_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(library_id: str, document_id: str, svc: VectorDBService = Depends(get_service)):
    ok = svc.delete_document(library_id, document_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document")
    return None
