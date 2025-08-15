from __future__ import annotations
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.core.storage import DiskStorage
from app.core.vector_db import VectorDBService
from app.api.routes import libraries, documents, chunks, search, embed

API_TITLE = "Vector Database API"
API_VERSION = os.getenv("API_VERSION", "0.1.0")
API_DOCS = os.getenv("API_DOCS", "/docs")
API_REDOC = os.getenv("API_REDOC", "/redoc")
API_OPENAPI = os.getenv("API_OPENAPI", "/openapi.json")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
ROOT_PATH = os.getenv("ROOT_PATH", "")
DATA_DIR = os.getenv("DATA_DIR", "db")  # where DiskStorage persists objects

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: initialize storage and service once."""
    log.info("Starting Vector Database API...")

    storage = DiskStorage(DATA_DIR) # persistent storage
    
    # Orchestrator: talks to storage and manages in-RAM indexes
    service = VectorDBService(storage)

    # TODO: load snapshots from disk on boot for fast init [work in progress]
    # for lib in service.list_libraries():
    #     service.rebuild_index(lib.id)

    app.state.storage = storage
    app.state.service = service

    try:
        yield
    finally:
        log.info("Shutting down Vector Database API...")


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
    api.include_router(chunks.router,    prefix="/libraries", tags=["Chunks"])
    api.include_router(search.router,    prefix="/libraries", tags=["Search"])
    api.include_router(embed.router,     prefix="",           tags=["Embeddings"])

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
        return {"message": "OK", "data_dir": getattr(svc.storage, "data_dir", DATA_DIR)}

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
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
        log_level=os.getenv("API_LOG_LEVEL", "info"),
        factory=False,
    )
