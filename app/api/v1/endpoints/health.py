from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.services.vector_store import vector_store_service
from app.core.logging import logger

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "RAG Document Analyzer AI Services",
        "version": "1.0.0"
    }


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check including all services."""
    try:
        # Check vector store health
        vector_store_health = await vector_store_service.health_check()
        
        # Determine overall health
        overall_status = "healthy"
        if vector_store_health.get("status") != "healthy":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "services": {
                "vector_store": vector_store_health
            },
            "service": "RAG Document Analyzer AI Services",
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@router.get("/vector-store")
async def vector_store_health() -> Dict[str, Any]:
    """Check vector store specific health."""
    try:
        return await vector_store_service.health_check()
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e)
            }
        ) 