import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import uvicorn

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.logging import setup_logging, log_request, log_error, logger
from app.services.vector_store import vector_store_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG Document Analyzer AI Services...")
    
    try:
        # Initialize logging
        setup_logging()
        logger.info("Logging configured")
        
        # Initialize vector store service
        await vector_store_service.initialize()
        logger.info("Vector store service initialized")
        
        # Application is ready
        logger.info("AI Services started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start AI Services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Services...")


# Create FastAPI application
app = FastAPI(
    title="RAG Document Analyzer AI Services",
    description="AI services for document analysis using RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENVIRONMENT == "development" else [settings.BACKEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", settings.HOST]
    )


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all HTTP requests."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        log_request(
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration=duration
        )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        log_error(e, {
            "method": request.method,
            "url": str(request.url),
            "duration": duration
        })
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    log_error(
        Exception(f"HTTP {exc.status_code}: {exc.detail}"),
        {
            "method": request.method,
            "url": str(request.url),
            "status_code": exc.status_code
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    log_error(exc, {
        "method": request.method,
        "url": str(request.url)
    })
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url.path)
        }
    )


# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "RAG Document Analyzer AI Services",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if settings.ENVIRONMENT != "production" else "disabled in production"
    }


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "service": "RAG Document Analyzer AI Services",
        "api_version": "v1",
        "endpoints": {
            "health": "/api/v1/health",
            "documents": "/api/v1/documents",
            "query": "/api/v1/query"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower()
    )