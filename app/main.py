from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.core.mongo_storage import MongoStorage
from app.core.vector_db import VectorDBService
from app.api.routes import libraries, documents, chunks, operations, embed

API_TITLE = "Vector Database API"
API_VERSION = os.getenv("API_VERSION", "0.1.0")
API_DOCS = os.getenv("API_DOCS", "/docs")
API_REDOC = os.getenv("API_REDOC", "/redoc")
API_OPENAPI = os.getenv("API_OPENAPI", "/openapi.json")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
ROOT_PATH = os.getenv("ROOT_PATH", "")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "vector_db")

# Check if we're in test mode
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
if TEST_MODE:
    MONGODB_DB = "test"
    print(f"🧪 TEST MODE: Using database '{MONGODB_DB}'")

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: initialize storage and service once."""
    log.info("Starting Vector Database API...")
    
    storage = MongoStorage(MONGODB_URI, MONGODB_DB) # persistent storage
    await storage._create_indexes()
    
    service = VectorDBService(storage) # Orchestrator: talks to storage and manages in-RAM indexes


    app.state.storage = storage
    app.state.service = service

    try:
        yield
    finally:
        log.info("Shutting down Vector Database API...")
        await storage.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        description="A custom-built Vector Database with a REST API",
        version=API_VERSION,
        docs_url=API_DOCS,
        redoc_url=API_REDOC,
        openapi_url=API_OPENAPI,
        root_path=ROOT_PATH,
        lifespan=lifespan,
    )

    # Classic middlewares
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    api = APIRouter(prefix="/v1")

    # sub-routers under /v1
    api.include_router(libraries.router, prefix="/libraries", tags=["Libraries"])
    api.include_router(documents.router, prefix="/libraries", tags=["Documents"])
    api.include_router(chunks.router, prefix="/libraries", tags=["Chunks"])
    api.include_router(operations.router, prefix="/libraries", tags=["Operations"])
    api.include_router(embed.router, prefix="", tags=["Embeddings"])

    app.include_router(api)

    # Health & version
    @app.get("/v1/healthz", status_code=status.HTTP_200_OK)
    async def health_check():
        return {"status": "healthy", "message": "Vector Database API is running"}

    @app.get("/v1/version", status_code=status.HTTP_200_OK)
    async def get_version():
        return {"version": API_VERSION}

    # Quick check to confirm storage/service are initialized
    @app.get("/v1/test")
    async def test_boot(request: Request):
        svc = getattr(request.app.state, "service", None)
        if not svc:
            return {"error": "service not initialized"}
        return {"message": "OK", "mongodb_uri": MONGODB_URI, "mongodb_db": MONGODB_DB}

    # Root
    @app.get("/", status_code=status.HTTP_200_OK)
    async def root():
        return {
            "message": API_TITLE,
            "version": API_VERSION,
            "docs": API_DOCS,
            "health": "/v1/healthz",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload="true",
        log_level="info",
        factory=False,
    )
