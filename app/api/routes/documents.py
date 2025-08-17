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
    """Create a new document within a library."""
    try:
        doc = await svc.create_document(library_id, body.title, body.metadata)
        return DocumentResponse(**doc.dict())
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found with the specified ID")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))



@router.get("/{library_id}/documents", response_model=List[DocumentResponse])
async def list_documents(library_id: str, svc: VectorDBService = Depends(get_service)):
    """List all documents within a library."""
    lib = await svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found with the specified ID")
    docs = await svc.list_documents(library_id)
    return [DocumentResponse(**d.dict()) for d in docs]



@router.get("/{library_id}/documents/{document_id}", response_model=DocumentResponse)
async def get_document(library_id: str, document_id: str, svc: VectorDBService = Depends(get_service)):
    """Get details of a specific document by ID."""
    lib = await svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found with the specified ID")
    doc = await svc.get_document(document_id)
    if not doc or doc.library_id != library_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found with the specified ID")
    return DocumentResponse(**doc.dict())



@router.patch("/{library_id}/documents/{document_id}", response_model=DocumentResponse)
async def update_document(library_id: str, document_id: str, body: UpdateDocumentRequest, svc: VectorDBService = Depends(get_service)):
    """Update document title or metadata."""
    try:
        # First verify the document exists
        doc = await svc.get_document(document_id)
        if not doc or doc.library_id != library_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found with the specified ID")
        
        updates = {}
        if body.title is not None: updates["title"] = body.title
        if body.metadata is not None: updates["metadata"] = body.metadata
        
        # if no updates, return original document
        if updates:
            try:
                updated = await svc.update_document(document_id, updates)
                if updated:
                    return DocumentResponse(**updated.dict())
            except ValueError as e:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        
        # No updates or update returned None (no changes needed)
        return DocumentResponse(**doc.dict())
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))



@router.delete("/{library_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(library_id: str, document_id: str, svc: VectorDBService = Depends(get_service)):
    """Delete a document and all its associated chunks."""
    ok = await svc.delete_document(library_id, document_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found with the specified ID")
    return None
