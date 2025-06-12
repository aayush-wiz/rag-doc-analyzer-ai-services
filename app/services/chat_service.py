# ai-service/app/services/chat_service.py
import logging
import time
from typing import Dict, Any

from app.core.query_engine import RAGQueryEngine
from app.core.vector_store import VectorStoreManager
from app.models.requests import ChatQueryRequest, ChatSummaryRequest
from app.models.responses import (
    ChatQueryResponse,
    ChatSummaryResponse,
    SourceDocument,
    QueryMetadata,
)

logger = logging.getLogger(__name__)


class ChatService:
    """Business logic for chat and query operations"""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.query_engine = RAGQueryEngine(vector_store_manager)

    async def process_query(self, request: ChatQueryRequest) -> ChatQueryResponse:
        """Process a RAG query and return response"""
        start_time = time.time()

        logger.info(
            f"Processing query for chat {request.chat_id}: {request.query[:100]}..."
        )

        try:
            # Execute RAG query
            result = await self.query_engine.query(
                query_text=request.query,
                chat_id=request.chat_id,
                max_results=request.max_results,
                document_filter=request.document_filter,
            )

            processing_time = time.time() - start_time

            # Convert sources to response model
            sources = [
                SourceDocument(
                    document_id=source["document_id"],
                    file_name=source["file_name"],
                    file_type=source["file_type"],
                    relevance_score=source["relevance_score"],
                    excerpt=source["excerpt"],
                )
                for source in result["sources"]
            ]

            # Create metadata
            metadata = QueryMetadata(
                total_documents_searched=result["metadata"]["total_documents_searched"],
                relevant_chunks_found=result["metadata"]["relevant_chunks_found"],
                query_length=result["metadata"]["query_length"],
                response_length=result["metadata"]["response_length"],
                error=result["metadata"].get("error"),
            )

            response = ChatQueryResponse(
                answer=result["answer"],
                sources=sources,
                confidence=result["confidence"],
                metadata=metadata,
                processing_time=round(processing_time, 2),
            )

            logger.info(
                f"Query processed for chat {request.chat_id} in {processing_time:.2f}s"
            )
            return response

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing query: {str(e)}"

            logger.error(
                f"Query processing error for chat {request.chat_id}: {error_msg}"
            )

            # Return error response
            return ChatQueryResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence=0.0,
                metadata=QueryMetadata(
                    total_documents_searched=0,
                    relevant_chunks_found=0,
                    query_length=len(request.query),
                    response_length=0,
                    error=error_msg,
                ),
                processing_time=round(processing_time, 2),
            )

    async def generate_summary(
        self, request: ChatSummaryRequest
    ) -> ChatSummaryResponse:
        """Generate summary of all documents in a chat"""
        start_time = time.time()

        logger.info(f"Generating summary for chat {request.chat_id}")

        try:
            # Generate chat summary
            result = await self.query_engine.get_chat_summary(request.chat_id)

            processing_time = time.time() - start_time

            response = ChatSummaryResponse(
                summary=result["summary"],
                key_topics=result["key_topics"],
                document_count=result["document_count"],
                total_chunks=result.get("total_chunks", 0),
                processing_time=round(processing_time, 2),
            )

            logger.info(
                f"Summary generated for chat {request.chat_id} in {processing_time:.2f}s"
            )
            return response

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error generating summary: {str(e)}"

            logger.error(
                f"Summary generation error for chat {request.chat_id}: {error_msg}"
            )

            return ChatSummaryResponse(
                summary=f"Error generating summary: {str(e)}",
                key_topics=[],
                document_count=0,
                total_chunks=0,
                processing_time=round(processing_time, 2),
            )

    async def get_chat_stats(self, chat_id: int) -> Dict[str, Any]:
        """Get statistics for a chat"""
        try:
            collection_name = f"chat_{chat_id}"
            stats = await self.vector_store_manager.get_collection_stats(
                collection_name
            )
            return stats

        except Exception as e:
            logger.error(f"Error getting chat stats for {chat_id}: {str(e)}")
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "collection_name": f"chat_{chat_id}",
                "error": str(e),
            }

    async def clear_chat(self, chat_id: int) -> bool:
        """Clear all documents from a chat"""
        try:
            collection_name = f"chat_{chat_id}"
            success = await self.vector_store_manager.delete_collection(collection_name)

            if success:
                logger.info(f"Cleared all documents from chat {chat_id}")
            else:
                logger.warning(f"Failed to clear documents from chat {chat_id}")

            return success

        except Exception as e:
            logger.error(f"Error clearing chat {chat_id}: {str(e)}")
            return False
