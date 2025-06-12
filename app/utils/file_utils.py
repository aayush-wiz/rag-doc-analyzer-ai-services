# ai-service/app/utils/file_utils.py
import logging
import asyncio
from typing import Optional
import boto3
from botocore.exceptions import ClientError
import mimetypes
from pathlib import Path

from app.config.settings import settings

logger = logging.getLogger(__name__)

# Initialize R2 client
s3_client = boto3.client(
    "s3",
    endpoint_url=settings.r2_endpoint_url,
    aws_access_key_id=settings.R2_ACCESS_KEY_ID,
    aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
    region_name="auto",
)


async def download_file_from_r2(file_path: str) -> bytes:
    """
    Download file from Cloudflare R2 storage

    Args:
        file_path: R2 object key or full URL

    Returns:
        File content as bytes
    """
    try:
        # Extract key from URL if needed
        if file_path.startswith("http"):
            # Extract key from URL
            key = file_path.split(f"{settings.R2_BUCKET}/")[-1]
        else:
            key = file_path

        logger.info(f"Downloading file from R2: {key}")

        # Download file in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: s3_client.get_object(Bucket=settings.R2_BUCKET, Key=key)
        )

        file_content = response["Body"].read()
        logger.info(f"Downloaded {len(file_content)} bytes from R2")

        return file_content

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchKey":
            raise FileNotFoundError(f"File not found in R2: {file_path}")
        elif error_code == "AccessDenied":
            raise PermissionError(f"Access denied to R2 file: {file_path}")
        else:
            raise Exception(f"R2 error downloading file {file_path}: {str(e)}")

    except Exception as e:
        logger.error(f"Error downloading file from R2: {str(e)}")
        raise Exception(f"Failed to download file from R2: {str(e)}")


async def upload_file_to_r2(
    file_content: bytes, key: str, content_type: str = None
) -> str:
    """
    Upload file to Cloudflare R2 storage

    Args:
        file_content: File content as bytes
        key: R2 object key
        content_type: MIME type of file

    Returns:
        Public URL of uploaded file
    """
    try:
        if not content_type:
            content_type = get_content_type(key)

        logger.info(f"Uploading file to R2: {key}")

        # Upload file in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.put_object(
                Bucket=settings.R2_BUCKET,
                Key=key,
                Body=file_content,
                ContentType=content_type,
            ),
        )

        # Generate public URL
        url = f"{settings.r2_endpoint_url}/{settings.R2_BUCKET}/{key}"
        logger.info(f"File uploaded to R2: {url}")

        return url

    except Exception as e:
        logger.error(f"Error uploading file to R2: {str(e)}")
        raise Exception(f"Failed to upload file to R2: {str(e)}")


def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return Path(filename).suffix.lower()


def get_content_type(filename: str) -> str:
    """Get MIME type from filename"""
    content_type, _ = mimetypes.guess_type(filename)
    return content_type or "application/octet-stream"


def validate_file_extension(filename: str) -> bool:
    """Validate if file extension is supported"""
    extension = get_file_extension(filename)
    return extension in settings.SUPPORTED_EXTENSIONS


def validate_file_size(file_size: int) -> bool:
    """Validate if file size is within limits"""
    return file_size <= settings.MAX_FILE_SIZE


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re

    # Remove dangerous characters
    sanitized = re.sub(r"[^\w\-_\.]", "_", filename)
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
        sanitized = name[:250] + ("." + ext if ext else "")
    return sanitized
