# ai-service/app/api/middleware/auth.py
import logging
from typing import Optional, Dict, Any
import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config.settings import settings

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()


async def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """
    Verify JWT token and extract user information

    Args:
        credentials: Bearer token from Authorization header

    Returns:
        User information from token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    token = credentials.credentials

    try:
        # Decode JWT token
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])

        # Extract user ID
        user_id = payload.get("id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Return user information
        return {"id": user_id, "exp": payload.get("exp"), "iat": payload.get("iat")}

    except jwt.ExpiredSignatureError:
        logger.warning(f"Expired JWT token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def verify_optional_jwt_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """
    Verify JWT token optionally (for endpoints that work with or without auth)

    Args:
        credentials: Optional bearer token

    Returns:
        User information if token provided and valid, None otherwise
    """
    if not credentials:
        return None

    try:
        return await verify_jwt_token(credentials)
    except HTTPException:
        return None


def create_service_token() -> str:
    """
    Create internal service token for backend communication

    Returns:
        JWT token for service-to-service communication
    """
    import time

    payload = {
        "service": "ai-service",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,  # 1 hour expiry
    }

    token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")
    return token


def verify_service_token(token: str) -> bool:
    """
    Verify internal service token

    Args:
        token: JWT token to verify

    Returns:
        True if token is valid service token
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        return payload.get("service") == "ai-service"
    except:
        return False
