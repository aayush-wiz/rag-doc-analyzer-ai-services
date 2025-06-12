# ai-service/app/config/settings.py (UPDATED for Gemini/Claude)
from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""

    # Environment
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Security
    JWT_SECRET: str
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:3001",
    ]
    ALLOWED_HOSTS: List[str] = ["*"]

    # AI/LLM Configuration - Gemini & Claude
    GEMINI_API_KEY: str
    CLAUDE_API_KEY: str

    # Primary LLM (gemini or claude)
    PRIMARY_LLM: str = "gemini"
    FALLBACK_LLM: str = "claude"

    # Gemini Configuration
    GEMINI_MODEL: str = "gemini-1.5-pro"
    GEMINI_EMBEDDING_MODEL: str = "models/embedding-001"

    # Claude Configuration
    CLAUDE_MODEL: str = "claude-3-sonnet-20241022"

    # General LLM settings
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 2000

    # Embedding Configuration
    EMBEDDING_MODEL_TYPE: str = (
        "sentence-transformers"  # sentence-transformers or gemini
    )
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

    # ChromaDB Configuration
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_COLLECTION_NAME: str = "document_embeddings"
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"

    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    SUPPORTED_EXTENSIONS: List[str] = [
        ".pdf",
        ".docx",
        ".doc",
        ".txt",
        ".csv",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
    ]

    # Storage Configuration (Cloudflare R2)
    R2_ACCOUNT_ID: str
    R2_ACCESS_KEY_ID: str
    R2_SECRET_ACCESS_KEY: str
    R2_BUCKET: str
    R2_ENDPOINT: Optional[str] = None

    # Backend Integration
    BACKEND_URL: str = "http://localhost:3001"
    BACKEND_API_TIMEOUT: int = 30

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000

    # Caching
    CACHE_TTL: int = 3600  # 1 hour
    REDIS_URL: Optional[str] = None

    # Vector Search
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_SEARCH_RESULTS: int = 10

    @property
    def r2_endpoint_url(self) -> str:
        """Generate R2 endpoint URL"""
        if self.R2_ENDPOINT:
            return self.R2_ENDPOINT
        return f"https://{self.R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

    @property
    def chroma_client_settings(self) -> dict:
        """ChromaDB client settings"""
        if self.ENVIRONMENT == "development":
            return {"path": self.CHROMA_PERSIST_DIRECTORY}
        else:
            return {"host": self.CHROMA_HOST, "port": self.CHROMA_PORT}

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create settings instance
settings = Settings()


# Validate required settings
def validate_settings():
    """Validate that all required settings are present"""
    required_settings = [
        "JWT_SECRET",
        "GEMINI_API_KEY",
        "CLAUDE_API_KEY",
        "R2_ACCOUNT_ID",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
    ]

    missing = []
    for setting in required_settings:
        if not getattr(settings, setting, None):
            missing.append(setting)

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


# Validate on import
validate_settings()
