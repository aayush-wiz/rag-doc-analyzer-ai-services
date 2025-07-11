from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from config import config
import re


class ValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            # Validate chat_id format
            body = await request.json()
            chat_id = body.get("chat_id")
            if chat_id and not re.match(r"^[a-zA-Z0-9_-]{1,50}$", chat_id):
                raise HTTPException(status_code=400, detail="Invalid chat_id format")

            # Validate request size
            content_length = int(request.headers.get("content-length", 0))
            if content_length > 50 * 1024 * 1024:  # 50MB limit
                raise HTTPException(status_code=413, detail="Request body too large")

        response = await call_next(request)
        return response
