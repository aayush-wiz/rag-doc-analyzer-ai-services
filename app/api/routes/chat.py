# ai-service/app/api/routes/chat.py
from fastapi import APIRouter, Depends, HTTPException
import logging

from app.models.requests import ChatQueryRequest, ChatSummaryRequest
from app.models.responses import ChatQueryResponse, ChatSummaryResponse
from app.services.chat_service import ChatService
from app.api.middleware.auth import verify_jwt_token
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)
router = APIRouter()


def get_vector_store_manager() -> VectorStoreManager:
    """Dependency to get vector store manager from app state"""
    from fastapi import Request

    # This will be injected by FastAPI
    return None


def get_chat_service(
    vector_store_manager: VectorStoreManager = Depends(get_vector_store_manager),
) -> ChatService:
    """Dependency to get chat service"""
    return ChatService(vector_store_manager)


@router.post("/query", response_model=ChatQueryResponse)
async def process_query(
    request: ChatQueryRequest,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: dict = Depends(verify_jwt_token),
):
    """
    Process a RAG query against uploaded documents

    This endpoint takes a user query, searches for relevant document chunks,
    and generates an AI response based on the retrieved context.
    """
    try:
        logger.info(
            f"Query request from user {current_user.get('id')} for chat {request.chat_id}"
        )

        result = await chat_service.process_query(request)
        return result

    except Exception as e:
        logger.error(f"Error in process_query endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


@router.post("/summary", response_model=ChatSummaryResponse)
async def generate_summary(
    request: ChatSummaryRequest,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: dict = Depends(verify_jwt_token),
):
    """
    Generate a summary of all documents in a chat

    This endpoint analyzes all uploaded documents and generates
    a comprehensive summary with key topics.
    """
    try:
        logger.info(
            f"Summary request from user {current_user.get('id')} for chat {request.chat_id}"
        )

        result = await chat_service.generate_summary(request)
        return result

    except Exception as e:
        logger.error(f"Error in generate_summary endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Summary generation failed: {str(e)}"
        )


@router.get("/stats/{chat_id}")
async def get_chat_stats(
    chat_id: int,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: dict = Depends(verify_jwt_token),
):
    """Get chat statistics"""
    try:
        stats = await chat_service.get_chat_stats(chat_id)
        return stats

    except Exception as e:
        logger.error(f"Error getting chat stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get chat stats: {str(e)}"
        )


@router.delete("/clear/{chat_id}")
async def clear_chat(
    chat_id: int,
    chat_service: ChatService = Depends(get_chat_service),
    current_user: dict = Depends(verify_jwt_token),
):
    """Clear all documents from a chat"""
    try:
        success = await chat_service.clear_chat(chat_id)

        return {
            "success": success,
            "message": (
                f"Chat {chat_id} cleared successfully"
                if success
                else f"Failed to clear chat {chat_id}"
            ),
        }

    except Exception as e:
        logger.error(f"Error clearing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear chat: {str(e)}")
