import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict


class Config:
    """Centralized configuration management."""

    def __init__(self):
        load_dotenv()
        self.STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 20 * 1024 * 1024))  # 20MB
        self.SUPPORTED_FILE_TYPES = {
            ".txt": "text/plain",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".md": "text/markdown",
        }
        self.MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 50000))
        self.CHAT_MEMORY_TOKEN_LIMIT = int(os.getenv("CHAT_MEMORY_TOKEN_LIMIT", 5000))
        self.VECTOR_STORAGE_PATH = Path(self.STORAGE_DIR) / "vector_storage"
        self.HISTORY_STORAGE_PATH = Path(self.STORAGE_DIR) / "chat_history"

    def validate(self) -> bool:
        """Validates required environment variables."""
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env file")
        return True

    def ensure_directories(self):
        """Creates storage directories if they don't exist."""
        self.VECTOR_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
        self.HISTORY_STORAGE_PATH.mkdir(parents=True, exist_ok=True)


config = Config()
