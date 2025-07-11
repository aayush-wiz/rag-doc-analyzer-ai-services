import base64
import hashlib
from pathlib import Path
from typing import Tuple
import mimetypes


def decode_base64(content: str) -> bytes:
    """Decodes base64 string to bytes."""
    try:
        return base64.b64decode(content)
    except Exception as e:
        raise ValueError(f"Invalid base64 content: {str(e)}")


def validate_file(
    file_name: str, file_content: bytes, max_size: int, supported_types: dict
) -> Tuple[bool, str]:
    """Validates file type and size."""
    file_extension = Path(file_name).suffix.lower()
    if file_extension not in supported_types:
        return False, f"Unsupported file type: {file_extension}"

    mime_type, _ = mimetypes.guess_type(file_name)
    if mime_type not in supported_types.values():
        return False, f"Invalid MIME type for {file_name}"

    if len(file_content) > max_size:
        return False, f"File size exceeds {max_size / (1024 * 1024)}MB limit"

    return True, ""


def generate_file_hash(content: bytes) -> str:
    """Generates SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()
