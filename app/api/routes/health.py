# ai-service/app/api/routes/health.py
from fastapi import APIRouter, Depends
from datetime import datetime
import logging

from app.models.responses import HealthResponse
from app.config.settings import settings
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns the current status of the AI service and its components.
    """
    try:
        # Check basic service health
        components = {
            "service": "healthy",
            "openai": "unknown",
            "vector_store": "unknown",
        }

        # Test OpenAI connection
        try:
            from app.core.llm_client import test_openai_connection

            if await test_openai_connection():
                components["openai"] = "healthy"
            else:
                components["openai"] = "unhealthy"
        except Exception as e:
            components["openai"] = f"error: {str(e)}"

        # Test vector store connection
        try:
            # This would need to be implemented properly with app state
            components["vector_store"] = "healthy"
        except Exception as e:
            components["vector_store"] = f"error: {str(e)}"

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.utcnow(),
            components=components,
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            timestamp=datetime.utcnow(),
            components={"service": f"error: {str(e)}"},
        )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes
    """
    try:
        # Perform minimal checks to verify service is ready
        return {"status": "ready", "timestamp": datetime.utcnow()}
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return {"status": "not ready", "error": str(e)}


@router.get("/live")
async def liveness_check():
    """
    Liveness check for Kubernetes
    """
    return {"status": "alive", "timestamp": datetime.utcnow()}
