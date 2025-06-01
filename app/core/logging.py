import logging
import sys
from typing import Dict, Any
import structlog
from structlog.stdlib import LoggerFactory
from app.core.config import settings


def setup_logging() -> None:
    """Setup structured logging with structlog."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.ENVIRONMENT == "production" 
            else structlog.dev.ConsoleRenderer(colors=True)
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("llama_index").setLevel(logging.WARNING)


# Create a logger instance
logger = structlog.get_logger(__name__)


def log_request(method: str, url: str, status_code: int, duration: float) -> None:
    """Log HTTP request details."""
    logger.info(
        "HTTP Request",
        method=method,
        url=url,
        status_code=status_code,
        duration_ms=round(duration * 1000, 2)
    )


def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Log error with context."""
    logger.error(
        "Error occurred",
        error=str(error),
        error_type=type(error).__name__,
        context=context or {}
    ) 