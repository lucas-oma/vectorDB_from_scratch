from fastapi import Request, HTTPException, status
from app.core.vector_db import VectorDBService

def get_service(request: Request) -> VectorDBService:
    svc = getattr(request.app.state, "service", None)
    if not svc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Vector database service not initialized. Please check server configuration.")
    return svc
