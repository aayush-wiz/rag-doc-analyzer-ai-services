import os
from pydantic import BaseSettings, Field
from typing import List, Optional

class Settings(BaseSettings):
    # API Configuration
    HOST: str = Field("0.0.0.0", env="HOST")
    PORT: int = Field(8000, env="PORT")
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    OPENAI_EMBEDDING_MODEL: str = Field("text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")
    OPENAI_TEMPERATURE: float = Field(0.1, env="OPENAI_TEMPERATURE")
    OPENAI_MAX_TOKENS: int = Field(2000, env="OPENAI_MAX_TOKENS")
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str = Field("./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    CHROMA_COLLECTION_NAME: str = Field("documents", env="CHROMA_COLLECTION_NAME")
    CHROMA_HOST: Optional[str] = Field(None, env="CHROMA_HOST")
    CHROMA_PORT: Optional[int] = Field(None, env="CHROMA_PORT")
    
    # Document Processing Configuration
    CHUNK_SIZE: int = Field(1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(200, env="CHUNK_OVERLAP")
    MAX_FILE_SIZE_MB: int = Field(50, env="MAX_FILE_SIZE_MB")
    
    # RAG Configuration
    SIMILARITY_TOP_K: int = Field(5, env="SIMILARITY_TOP_K")
    RESPONSE_MODE: str = Field("tree_summarize", env="RESPONSE_MODE")
    
    # Security Configuration
    SECRET_KEY: str = Field("your-secret-key-change-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Backend Integration
    BACKEND_URL: str = Field("http://localhost:5001", env="BACKEND_URL")
    BACKEND_SECRET_KEY: str = Field("backend-secret", env="BACKEND_SECRET_KEY")
    
    # File Storage Configuration
    UPLOAD_DIRECTORY: str = Field("./uploads", env="UPLOAD_DIRECTORY")
    SUPPORTED_FILE_TYPES: list = [".pdf", ".txt", ".docx", ".md", ".xlsx", ".csv"]
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Monitoring Configuration
    PROMETHEUS_ENABLED: bool = Field(True, env="PROMETHEUS_ENABLED")
    PROMETHEUS_PORT: int = Field(9090, env="PROMETHEUS_PORT")
    
    # Optional: Redis Configuration (for caching)
    REDIS_URL: Optional[str] = Field(None, env="REDIS_URL")
    
    # Optional: Google Cloud Configuration
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(None, env="GOOGLE_APPLICATION_CREDENTIALS")
    GCS_BUCKET_NAME: Optional[str] = Field(None, env="GCS_BUCKET_NAME")
    FIRESTORE_PROJECT_ID: Optional[str] = Field(None, env="FIRESTORE_PROJECT_ID")
    
    # Optional: Gemini Configuration
    GEMINI_API_KEY: Optional[str] = Field(None, env="GEMINI_API_KEY")
    
    @property
    def supported_file_types_list(self) -> List[str]:
        return [ext.strip() for ext in self.SUPPORTED_FILE_TYPES]
    
    @property
    def chroma_db_url(self) -> str:
        return f"http://{self.CHROMA_HOST}:{self.CHROMA_PORT}" if self.CHROMA_HOST and self.CHROMA_PORT else "ChromaDB not configured"
    
    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Ensure required directories exist
os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True) 