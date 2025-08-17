from __future__ import annotations
from app.api.routes.embed import embed_texts
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_service
from app.core.vector_db import VectorDBService
from app.api.dto import (
    RebuildIndexRequest, TrainIndexRequest, SearchRequest, SearchTextRequest,
    SearchResponse, LibraryStatsResponse, ChunkResponse, EmbedRequest
)

router = APIRouter()

@router.post("/{library_id}/index/rebuild", status_code=status.HTTP_202_ACCEPTED)
async def rebuild_index(library_id: str, body: RebuildIndexRequest, svc: VectorDBService = Depends(get_service)):
    """Rebuild the vector index for a library from persistent data."""
    lib = await svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found with the specified ID")
    
    try:
        await svc.rebuild_index(library_id)
        return {"message": f"Index rebuilt for library '{library_id}'", "index_type": lib.index_type, "dims": lib.dims}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))



@router.post("/{library_id}/index/train", status_code=status.HTTP_202_ACCEPTED)
async def train_index(library_id: str, body: TrainIndexRequest, svc: VectorDBService = Depends(get_service)):
    """Train an index with sample vectors (required for IVF indexes)."""
    """Train an index. Only works for indexes that require training (example: IVF)."""
    try:
        # Use the service method with proper locking
        await svc.train_index(library_id, sample_vectors=body.sample_vectors)
        
        lib = await svc.get_library(library_id)
        return {
            "message": f"Index trained for library '{library_id}'", 
            "index_type": lib.index_type, 
            "dims": lib.dims,
        }
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found with the specified ID")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))



@router.get("/{library_id}/stats", response_model=LibraryStatsResponse)
async def get_library_stats(library_id: str, svc: VectorDBService = Depends(get_service)):
    """Get statistics about a library including document and chunk counts."""
    lib = await svc.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found with the specified ID")
    docs = await svc.list_documents(library_id)
    chs = await svc.list_chunks(library_id)
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
    """Search for similar chunks using a pre-computed embedding vector."""
    try:
        results = await svc.search(library_id, body.embedding, body.k, include_chunk=body.include_chunk)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found with the specified ID")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    out = []
    for r in results:
        chunk_resp = ChunkResponse(**r.chunk.dict()) if (body.include_chunk and r.chunk) else None
        out.append({"chunk_id": r.chunk_id, "similarity_score": r.similarity_score, "chunk": chunk_resp})

    return SearchResponse(library_id=library_id, results=out)



@router.post("/{library_id}/search_text", response_model=SearchResponse)
async def search_text(library_id: str, body: SearchTextRequest, svc: VectorDBService = Depends(get_service)):
    """Search for similar chunks using text query (auto-generates embedding)."""
    # Generate embedding from text
    try:
        embed_req = EmbedRequest(texts=[body.text])
        embed_response = await embed_texts(embed_req)
        embedding = embed_response.embeddings[0]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate embedding: {str(e)}")
    
    # Create SearchRequest and call regular search endpoint
    search_request = SearchRequest(
        embedding=embedding,
        k=body.k,
        include_chunk=body.include_chunk
    )
    
    return await search(library_id, search_request, svc)
