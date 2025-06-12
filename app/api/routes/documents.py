# ai-service/app/api/routes/documents.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from app.models.requests import DocumentProcessRequest, DocumentDeleteRequest
from app.models.responses import (
    DocumentProcessResponse,
    DocumentDeleteResponse,
    APIResponse,
)
from app.services.document_service import DocumentService
from app.api.middleware.auth import verify_jwt_token
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)
router = APIRouter()


def get_vector_store_manager() -> VectorStoreManager:
    """Dependency to get vector store manager from app state"""
    from fastapi import Request

    # This will be injected by FastAPI
    return None


def get_document_service(
    vector_store_manager: VectorStoreManager = Depends(get_vector_store_manager),
) -> DocumentService:
    """Dependency to get document service"""
    return DocumentService(vector_store_manager)


@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(
    request: DocumentProcessRequest,
    background_tasks: BackgroundTasks,
    document_service: DocumentService = Depends(get_document_service),
    current_user: dict = Depends(verify_jwt_token),
):
    """
    Process a document and store embeddings

    This endpoint processes uploaded documents, chunks them into smaller pieces,
    generates embeddings, and stores them in the vector database for retrieval.
    """
    try:
        logger.info(
            f"Document processing request from user {current_user.get('id')} for document {request.document_id}"
        )

        # Process document
        result = await document_service.process_document(request)

        return result

    except Exception as e:
        logger.error(f"Error in process_document endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Document processing failed: {str(e)}"
        )


@router.delete("/{chat_id}/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    chat_id: int,
    document_id: int,
    document_service: DocumentService = Depends(get_document_service),
    current_user: dict = Depends(verify_jwt_token),
):
    """
    Delete document embeddings from vector store

    This endpoint removes all embeddings and chunks for a specific document
    from the vector database.
    """
    try:
        request = DocumentDeleteRequest(chat_id=chat_id, document_id=document_id)
        result = await document_service.delete_document(request)

        return result

    except Exception as e:
        logger.error(f"Error in delete_document endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Document deletion failed: {str(e)}"
        )


@router.get("/stats/{chat_id}")
async def get_document_stats(
    chat_id: int,
    document_service: DocumentService = Depends(get_document_service),
    current_user: dict = Depends(verify_jwt_token),
):
    """Get document statistics for a chat"""
    try:
        stats = await document_service.get_document_stats(chat_id)
        return stats

    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get document stats: {str(e)}"
        )
