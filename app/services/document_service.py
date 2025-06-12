# ai-service/app/services/document_service.py
import logging
import asyncio
import time
from typing import Dict, Any

from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStoreManager
from app.models.requests import DocumentProcessRequest, DocumentDeleteRequest
from app.models.responses import (
    DocumentProcessResponse,
    DocumentDeleteResponse,
    ProcessingStats,
)

logger = logging.getLogger(__name__)


class DocumentService:
    """Business logic for document processing operations"""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.document_processor = DocumentProcessor(vector_store_manager)

    async def process_document(
        self, request: DocumentProcessRequest
    ) -> DocumentProcessResponse:
        """Process a document and store embeddings"""
        start_time = time.time()

        logger.info(f"Starting document processing for document {request.document_id}")

        try:
            # Process the document
            result = await self.document_processor.process_document(
                chat_id=request.chat_id,
                document_id=request.document_id,
                file_path=request.file_path,
                file_name=request.file_name,
                file_type=request.file_type,
            )

            processing_time = time.time() - start_time

            if result["success"]:
                stats = ProcessingStats(
                    total_documents=result["stats"]["total_documents"],
                    total_nodes=result["stats"]["total_nodes"],
                    total_characters=result["stats"]["total_characters"],
                    average_node_size=result["stats"]["average_node_size"],
                )

                response = DocumentProcessResponse(
                    success=True,
                    document_id=request.document_id,
                    message=result["message"],
                    stats=stats,
                    processing_time=round(processing_time, 2),
                )

                logger.info(
                    f"Document {request.document_id} processed successfully in {processing_time:.2f}s"
                )

            else:
                response = DocumentProcessResponse(
                    success=False,
                    document_id=request.document_id,
                    message=result["message"],
                    error=result["error"],
                    processing_time=round(processing_time, 2),
                )

                logger.error(
                    f"Document {request.document_id} processing failed: {result['error']}"
                )

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Unexpected error processing document: {str(e)}"

            logger.error(
                f"Document {request.document_id} processing error: {error_msg}"
            )

            return DocumentProcessResponse(
                success=False,
                document_id=request.document_id,
                message="Document processing failed",
                error=error_msg,
                processing_time=round(processing_time, 2),
            )

    async def delete_document(
        self, request: DocumentDeleteRequest
    ) -> DocumentDeleteResponse:
        """Delete document embeddings from vector store"""

        logger.info(
            f"Deleting document {request.document_id} from chat {request.chat_id}"
        )

        try:
            success = await self.document_processor.delete_document(
                chat_id=request.chat_id, document_id=request.document_id
            )

            if success:
                message = f"Document {request.document_id} deleted successfully"
                logger.info(message)
            else:
                message = f"Failed to delete document {request.document_id}"
                logger.warning(message)

            return DocumentDeleteResponse(
                success=success, document_id=request.document_id, message=message
            )

        except Exception as e:
            error_msg = f"Error deleting document: {str(e)}"
            logger.error(f"Document {request.document_id} deletion error: {error_msg}")

            return DocumentDeleteResponse(
                success=False,
                document_id=request.document_id,
                message="Document deletion failed",
                error=error_msg,
            )

    async def get_document_stats(self, chat_id: int) -> Dict[str, Any]:
        """Get document statistics for a chat"""
        try:
            stats = await self.document_processor.get_document_stats(chat_id)
            return stats

        except Exception as e:
            logger.error(f"Error getting document stats for chat {chat_id}: {str(e)}")
            return {"error": str(e), "total_documents": 0, "total_chunks": 0}
