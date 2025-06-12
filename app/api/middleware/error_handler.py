# ai-service/app/api/middleware/error_handler.py
import logging
import traceback
from typing import Union
from datetime import datetime
import json

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.models.responses import ErrorResponse

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        """Process request and handle any errors"""
        try:
            response = await call_next(request)
            return response

        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            logger.warning(f"HTTP Exception: {e.status_code} - {e.detail}")

            error_response = ErrorResponse(
                error=e.detail,
                error_type="HTTP_EXCEPTION",
                details={"status_code": e.status_code},
            )

            return JSONResponse(
                status_code=e.status_code,
                content=error_response.dict(),
                headers=getattr(e, "headers", None),
            )

        except ValidationError as e:
            # Handle Pydantic validation errors
            logger.warning(f"Validation error: {str(e)}")

            error_response = ErrorResponse(
                error="Validation failed",
                error_type="VALIDATION_ERROR",
                details={"validation_errors": e.errors()},
            )

            return JSONResponse(status_code=422, content=error_response.dict())

        except Exception as e:
            # Handle unexpected errors
            error_id = datetime.utcnow().timestamp()
            logger.error(
                f"Unexpected error [{error_id}]: {str(e)}\n{traceback.format_exc()}"
            )

            error_response = ErrorResponse(
                error="Internal server error",
                error_type="INTERNAL_ERROR",
                details={
                    "error_id": error_id,
                    "message": (
                        str(e)
                        if not isinstance(e, Exception)
                        else "An unexpected error occurred"
                    ),
                },
            )

            return JSONResponse(status_code=500, content=error_response.dict())


# Import ValidationError if using Pydantic
try:
    from pydantic import ValidationError
except ImportError:

    class ValidationError(Exception):
        def errors(self):
            return []


def create_error_response(
    error: Union[str, Exception],
    error_type: str = "UNKNOWN_ERROR",
    status_code: int = 500,
    details: dict = None,
) -> JSONResponse:
    """
    Create standardized error response

    Args:
        error: Error message or exception
        error_type: Type/category of error
        status_code: HTTP status code
        details: Additional error details

    Returns:
        JSONResponse with error information
    """
    error_message = str(error) if isinstance(error, Exception) else error

    error_response = ErrorResponse(
        error=error_message, error_type=error_type, details=details or {}
    )

    return JSONResponse(status_code=status_code, content=error_response.dict())


def log_error(error: Exception, context: dict = None, level: str = "error") -> None:
    """
    Log error with context information

    Args:
        error: Exception to log
        context: Additional context information
        level: Log level (error, warning, info)
    """
    log_method = getattr(logger, level, logger.error)

    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {},
    }

    log_method(f"Error occurred: {json.dumps(error_info, indent=2)}")

    # Log stack trace for errors
    if level == "error":
        logger.error(f"Stack trace:\n{traceback.format_exc()}")


# Custom exception classes
class AIServiceError(Exception):
    """Base exception for AI service errors"""

    def __init__(
        self, message: str, error_type: str = "AI_SERVICE_ERROR", details: dict = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


class DocumentProcessingError(AIServiceError):
    """Exception for document processing errors"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "DOCUMENT_PROCESSING_ERROR", details)


class VectorStoreError(AIServiceError):
    """Exception for vector store operations"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "VECTOR_STORE_ERROR", details)


class LLMError(AIServiceError):
    """Exception for LLM operations"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "LLM_ERROR", details)
