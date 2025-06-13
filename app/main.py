# ai-services/app/main.py (SIMPLIFIED WORKING VERSION)
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import asyncio
from typing import Dict, Any
from datetime import datetime
from app.core.document_processor import DocumentProcessor
from app.core.query_engine import RAGQueryEngine

# Your existing imports
from app.config.settings import settings, validate_settings
from app.core.llm_client import llm_client
from app.core.vector_store import VectorStoreManager

# Your updated models (only if they exist)
try:
    from app.models.requests import (
        DocumentProcessRequest,
        ChatQueryRequest,
        ChatSummaryRequest,
        DocumentDeleteRequest,
    )
    from app.models.responses import (
        DocumentProcessResponse,
        ChatQueryResponse,
        ChatSummaryResponse,
        DocumentDeleteResponse,
    )

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Models not available, using basic responses")

# Setup logging using your LOG_LEVEL setting
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper()))
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="DocAnalyzer AI Service",
    description="AI-powered document processing and RAG service",
    version="2.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)

# Add CORS middleware using your settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Add to dependency injection
async def get_document_processor(
    vector_store: VectorStoreManager = Depends(get_vector_store_manager),
) -> DocumentProcessor:
    """Get document processor dependency"""
    return DocumentProcessor(vector_store)


async def get_query_engine(
    vector_store: VectorStoreManager = Depends(get_vector_store_manager),
) -> RAGQueryEngine:
    """Get RAG query engine dependency"""
    return RAGQueryEngine(vector_store)


# Global state
app.state.vector_store_manager = None


@app.on_event("startup")
async def startup_event():
    """Application startup with your settings"""
    logger.info("🚀 Starting AI Service (Simplified Version)...")

    try:
        # Validate settings first
        validate_settings()
        logger.info("✅ Settings validated")

        # Initialize vector store manager with your settings
        vector_store_manager = VectorStoreManager()
        await vector_store_manager.initialize()

        # Store globally for dependency injection
        app.state.vector_store_manager = vector_store_manager

        logger.info(f"✅ Vector store initialized")
        logger.info(
            f"✅ Primary LLM: {settings.PRIMARY_LLM}, Fallback: {settings.FALLBACK_LLM}"
        )
        logger.info(f"✅ Embedding model: {settings.EMBEDDING_MODEL_TYPE}")
        logger.info("✅ AI Service started successfully")

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("🛑 Shutting down AI Service...")

    if app.state.vector_store_manager:
        await app.state.vector_store_manager.close()

    logger.info("👋 AI Service stopped")


# Dependency injection
async def get_vector_store_manager() -> VectorStoreManager:
    """Get vector store manager dependency"""
    if not app.state.vector_store_manager:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    return app.state.vector_store_manager


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "error": "Internal server error",
                "error_type": "INTERNAL_ERROR",
                "details": (
                    {"message": str(exc)}
                    if settings.ENVIRONMENT == "development"
                    else None
                ),
                "timestamp": datetime.utcnow().isoformat(),
            },
        },
    )


# Enhanced health check using your settings
@app.get("/health")
async def health_check():
    """Enhanced health check with your configuration"""
    try:
        # Test vector store
        vector_store_status = "not_initialized"
        if app.state.vector_store_manager:
            try:
                await app.state.vector_store_manager.get_collection_stats(
                    "health_check"
                )
                vector_store_status = "connected"
            except Exception:
                vector_store_status = "error"

        # Check your LLM configurations
        gemini_status = "configured" if settings.GEMINI_API_KEY else "not_configured"
        claude_status = "configured" if settings.CLAUDE_API_KEY else "not_configured"

        # Check your storage
        r2_status = (
            "configured"
            if all(
                [
                    settings.R2_ACCOUNT_ID,
                    settings.R2_ACCESS_KEY_ID,
                    settings.R2_SECRET_ACCESS_KEY,
                    settings.R2_BUCKET,
                ]
            )
            else "not_configured"
        )

        return {
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "vector_store": vector_store_status,
                "primary_llm": f"{settings.PRIMARY_LLM} ({gemini_status if settings.PRIMARY_LLM == 'gemini' else claude_status})",
                "fallback_llm": f"{settings.FALLBACK_LLM} ({claude_status if settings.FALLBACK_LLM == 'claude' else gemini_status})",
                "embedding_model": settings.EMBEDDING_MODEL_TYPE,
                "r2_storage": r2_status,
                "environment": settings.ENVIRONMENT,
            },
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


# Root endpoint showing your configuration
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "DocAnalyzer AI Service",
        "version": "2.0.0",
        "architecture": "Simplified (Direct ChromaDB)",
        "configuration": {
            "primary_llm": settings.PRIMARY_LLM,
            "fallback_llm": settings.FALLBACK_LLM,
            "embedding_model": settings.EMBEDDING_MODEL_TYPE,
            "max_file_size": f"{settings.MAX_FILE_SIZE // (1024*1024)}MB",
            "supported_extensions": len(settings.SUPPORTED_EXTENSIONS),
            "environment": settings.ENVIRONMENT,
        },
        "status": "running",
        "docs": "/docs" if settings.ENVIRONMENT == "development" else "disabled",
    }


@app.post("/api/documents/process")
async def process_document(
    request: Dict[str, Any],
    doc_processor: DocumentProcessor = Depends(get_document_processor),
):
    """Process document endpoint (REAL IMPLEMENTATION)"""
    logger.info(
        f"Processing document {request.get('document_id')} for chat {request.get('chat_id')}"
    )

    try:
        result = await doc_processor.process_document(
            chat_id=request.get("chat_id"),
            document_id=request.get("document_id"),
            file_path=request.get("file_path"),
            file_name=request.get("file_name"),
            file_type=request.get("file_type"),
        )

        return {
            "success": result["success"],
            "document_id": request.get("document_id"),
            "message": result["message"],
            "stats": result.get("stats"),
            "error": result.get("error"),
            "processing_time": 1.0,  # You can add timing
        }

    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        return {
            "success": False,
            "document_id": request.get("document_id"),
            "message": "Document processing failed",
            "error": str(e),
            "processing_time": 0.0,
        }


@app.post("/api/documents/process")
async def process_document(
    request: Dict[str, Any],
    doc_processor: DocumentProcessor = Depends(get_document_processor),
):
    """Process document endpoint (REAL IMPLEMENTATION)"""
    logger.info(
        f"Processing document {request.get('document_id')} for chat {request.get('chat_id')}"
    )

    try:
        result = await doc_processor.process_document(
            chat_id=request.get("chat_id"),
            document_id=request.get("document_id"),
            file_path=request.get("file_path"),
            file_name=request.get("file_name"),
            file_type=request.get("file_type"),
        )

        return {
            "success": result["success"],
            "document_id": request.get("document_id"),
            "message": result["message"],
            "stats": result.get("stats"),
            "error": result.get("error"),
            "processing_time": 1.0,  # You can add timing
        }

    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        return {
            "success": False,
            "document_id": request.get("document_id"),
            "message": "Document processing failed",
            "error": str(e),
            "processing_time": 0.0,
        }


# Delete document endpoint
@app.delete("/api/documents/{chat_id}/{document_id}")
async def delete_document(
    chat_id: int,
    document_id: int,
    vector_store: VectorStoreManager = Depends(get_vector_store_manager),
):
    """Delete document endpoint"""
    try:
        collection_name = f"chat_{chat_id}"
        success = await vector_store.delete_document(collection_name, document_id)

        return {
            "success": success,
            "document_id": document_id,
            "message": (
                "Document deleted successfully"
                if success
                else "Document deletion failed"
            ),
        }

    except Exception as e:
        logger.error(f"Document deletion error: {str(e)}")
        return {
            "success": False,
            "document_id": document_id,
            "message": "Document deletion failed",
            "error": str(e),
        }


# Statistics endpoints
@app.get("/api/documents/stats/{chat_id}")
async def get_document_stats(
    chat_id: int, vector_store: VectorStoreManager = Depends(get_vector_store_manager)
):
    """Get document statistics"""
    collection_name = f"chat_{chat_id}"
    return await vector_store.get_collection_stats(collection_name)


@app.get("/api/chat/stats/{chat_id}")
async def get_chat_stats(
    chat_id: int, vector_store: VectorStoreManager = Depends(get_vector_store_manager)
):
    """Get chat statistics"""
    collection_name = f"chat_{chat_id}"
    return await vector_store.get_collection_stats(collection_name)


# Service configuration endpoint (showing your settings)
@app.get("/api/config")
async def get_configuration():
    """Get service configuration (non-sensitive)"""
    return {
        "llm": {
            "primary": settings.PRIMARY_LLM,
            "fallback": settings.FALLBACK_LLM,
            "temperature": settings.TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS,
        },
        "document_processing": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "max_file_size": settings.MAX_FILE_SIZE,
            "supported_extensions": settings.SUPPORTED_EXTENSIONS,
        },
        "vector_search": {
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "max_search_results": settings.MAX_SEARCH_RESULTS,
        },
        "rate_limits": {
            "per_minute": settings.RATE_LIMIT_PER_MINUTE,
            "per_hour": settings.RATE_LIMIT_PER_HOUR,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
