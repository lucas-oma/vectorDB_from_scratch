from __future__ import annotations
import os
import httpx
from fastapi import APIRouter, HTTPException, status

from app.api.dto import EmbedRequest, EmbedResponse

router = APIRouter()

COHERE_API_URL = os.getenv("COHERE_EMBED_URL", "https://api.cohere.ai/v1/embed")
COHERE_MODEL_DEFAULT = os.getenv("COHERE_MODEL", "embed-english-v3.0")
COHERE_TIMEOUT_S = float(os.getenv("COHERE_TIMEOUT_S", "10"))
EMBED_MAX_TEXTS = int(os.getenv("EMBED_MAX_TEXTS", "128"))

@router.post("/embed", response_model=EmbedResponse, status_code=status.HTTP_200_OK)
async def embed_texts(req: EmbedRequest) -> EmbedResponse:
    """Generate embedding vectors from text using the Cohere API."""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Cohere API key not configured. Please set COHERE_API_KEY environment variable.")

    texts = req.texts or []
    if not texts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Texts list cannot be empty. Please provide at least one text to embed.")
    if len(texts) > EMBED_MAX_TEXTS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Too many texts provided: {len(texts)} > {EMBED_MAX_TEXTS}. Please reduce the number of texts to embed.")

    payload = {
        "texts": texts,
        "model": req.model or COHERE_MODEL_DEFAULT,
        "input_type": "search_document",
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=COHERE_TIMEOUT_S) as client:
            resp = await client.post(COHERE_API_URL, headers=headers, json=payload)

        if 400 <= resp.status_code < 500:
            # Surface provider error content when possible
            try:
                err = resp.json()
            except Exception:
                err = {"message": resp.text}
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"provider_status": resp.status_code, "error": err})
        if resp.status_code >= 500:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Embedding provider (Cohere) server error: {resp.status_code}")

        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Embedding provider (Cohere) response is missing 'embeddings' field")
        return EmbedResponse(embeddings=embeddings)
    except httpx.TimeoutException:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Embedding provider (Cohere) request timed out. Please try again.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Embedding provider (Cohere) request failed: {str(e)}")

