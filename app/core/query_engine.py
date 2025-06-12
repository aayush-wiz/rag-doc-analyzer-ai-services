# ai-service/app/core/query_engine.py (UPDATED for Gemini/Claude)
import logging
from typing import List, Dict, Any, Optional
import asyncio

from llama_index.core.schema import NodeWithScore, BaseNode

from app.config.settings import settings
from app.core.vector_store import VectorStoreManager
from app.core.llm_client import llm_client

logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """RAG (Retrieval Augmented Generation) Query Engine with Gemini/Claude"""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager

    async def query(
        self,
        query_text: str,
        chat_id: int,
        max_results: int = None,
        document_filter: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Execute RAG query against chat documents
        """
        logger.info(f"Processing query for chat {chat_id}: {query_text[:100]}...")

        try:
            # Generate query embedding
            query_embedding = await llm_client.generate_embedding(query_text)

            # Retrieve relevant documents
            retrieved_nodes = await self._retrieve_documents(
                query_embedding=query_embedding,
                chat_id=chat_id,
                max_results=max_results,
                document_filter=document_filter,
            )

            if not retrieved_nodes:
                return {
                    "answer": "I couldn't find any relevant information in the uploaded documents to answer your question. Please make sure your documents are uploaded and processed.",
                    "sources": [],
                    "confidence": 0.0,
                    "metadata": {
                        "total_documents_searched": 0,
                        "relevant_chunks_found": 0,
                    },
                }

            # Generate response using LLM
            response = await self._generate_response(query_text, retrieved_nodes)

            # Format sources
            sources = self._format_sources(retrieved_nodes)

            # Calculate confidence score
            confidence = self._calculate_confidence(retrieved_nodes)

            result = {
                "answer": response,
                "sources": sources,
                "confidence": confidence,
                "metadata": {
                    "total_documents_searched": len(
                        set(
                            node.node.metadata.get("document_id")
                            for node in retrieved_nodes
                        )
                    ),
                    "relevant_chunks_found": len(retrieved_nodes),
                    "query_length": len(query_text),
                    "response_length": len(response),
                },
            }

            logger.info(
                f"Query completed for chat {chat_id}. Confidence: {confidence:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error processing query for chat {chat_id}: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}. Please try again or contact support if the issue persists.",
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "error": str(e),
                    "total_documents_searched": 0,
                    "relevant_chunks_found": 0,
                },
            }

    async def _retrieve_documents(
        self,
        query_embedding: List[float],
        chat_id: int,
        max_results: int = None,
        document_filter: Optional[List[int]] = None,
    ) -> List[NodeWithScore]:
        """Retrieve relevant documents from vector store"""

        collection_name = f"chat_{chat_id}"
        max_results = max_results or settings.MAX_SEARCH_RESULTS

        # Build filter if document IDs specified
        where_filter = None
        if document_filter:
            where_filter = {
                "document_id": {"$in": [str(doc_id) for doc_id in document_filter]}
            }

        # Search vector store
        nodes_with_scores = await self.vector_store_manager.search_similar(
            query_embedding=query_embedding,
            collection_name=collection_name,
            n_results=max_results,
            where_filter=where_filter,
        )

        # Filter by similarity threshold
        filtered_nodes = [
            node_score
            for node_score in nodes_with_scores
            if node_score.score >= settings.SIMILARITY_THRESHOLD
        ]

        logger.info(f"Retrieved {len(filtered_nodes)} relevant chunks for query")
        return filtered_nodes

    async def _generate_response(
        self, query_text: str, retrieved_nodes: List[NodeWithScore]
    ) -> str:
        """Generate response using LLM and retrieved context"""

        # Prepare context from retrieved nodes
        context_pieces = []
        for i, node_with_score in enumerate(retrieved_nodes[:5]):  # Use top 5 chunks
            node = node_with_score.node
            score = node_with_score.score
            source = node.metadata.get("file_name", "Unknown")

            context_pieces.append(
                f"[Source {i+1}: {source} (relevance: {score:.2f})]\n{node.text}\n"
            )

        context = "\n".join(context_pieces)

        # Create system prompt
        system_prompt = (
            """You are a helpful AI assistant that answers questions based on provided documents. 

Instructions:
1. Answer the user's question using ONLY the information provided in the context
2. If the context doesn't contain enough information to answer the question, say so
3. Always cite which source(s) you're using in your answer
4. Be concise but thorough
5. If multiple sources contradict each other, mention this

Context from documents:
"""
            + context
        )

        # Create user prompt
        user_prompt = f"Question: {query_text}\n\nPlease answer this question based on the provided context."

        # Generate response using LLM
        try:
            response = await llm_client.generate_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
            )
            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Try with fallback LLM
            try:
                response = await llm_client.generate_completion(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=settings.MAX_TOKENS,
                    temperature=settings.TEMPERATURE,
                    use_fallback=True,
                )
                return response
            except Exception as fallback_error:
                logger.error(f"Fallback LLM also failed: {str(fallback_error)}")
                return f"I apologize, but I'm having technical difficulties generating a response. Please try again later. Error: {str(e)}"

    def _format_sources(
        self, nodes_with_scores: List[NodeWithScore]
    ) -> List[Dict[str, Any]]:
        """Format source information from retrieved nodes"""
        sources = []
        seen_docs = set()

        for node_with_score in nodes_with_scores:
            node = node_with_score.node
            metadata = node.metadata

            doc_id = metadata.get("document_id")
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)

                source = {
                    "document_id": doc_id,
                    "file_name": metadata.get("file_name", "Unknown"),
                    "file_type": metadata.get("file_type", "Unknown"),
                    "relevance_score": round(node_with_score.score, 3),
                    "excerpt": (
                        node.text[:200] + "..." if len(node.text) > 200 else node.text
                    ),
                }
                sources.append(source)

        # Sort by relevance score
        sources.sort(key=lambda x: x["relevance_score"], reverse=True)
        return sources

    def _calculate_confidence(self, nodes_with_scores: List[NodeWithScore]) -> float:
        """Calculate confidence score based on retrieved results"""
        if not nodes_with_scores:
            return 0.0

        # Calculate weighted average of similarity scores
        total_score = sum(node.score for node in nodes_with_scores)
        avg_score = total_score / len(nodes_with_scores)

        # Boost confidence based on number of relevant results
        result_bonus = min(len(nodes_with_scores) / settings.MAX_SEARCH_RESULTS, 0.2)

        # Final confidence score (0.0 to 1.0)
        confidence = min(avg_score + result_bonus, 1.0)

        return round(confidence, 3)

    async def get_chat_summary(self, chat_id: int) -> Dict[str, Any]:
        """Generate summary of all documents in a chat"""
        try:
            # Get all nodes for the chat (limit to reduce processing time)
            collection_name = f"chat_{chat_id}"

            # Use a broad query to get representative content
            summary_query = (
                "What are the main topics and key information in these documents?"
            )
            query_embedding = await llm_client.generate_embedding(summary_query)

            nodes_with_scores = await self.vector_store_manager.search_similar(
                query_embedding=query_embedding,
                collection_name=collection_name,
                n_results=20,  # Get more nodes for summary
            )

            if not nodes_with_scores:
                return {
                    "summary": "No documents have been processed yet.",
                    "key_topics": [],
                    "document_count": 0,
                }

            # Generate summary
            summary_response = await self._generate_summary_response(nodes_with_scores)

            # Extract key topics (simple keyword extraction)
            key_topics = self._extract_key_topics(nodes_with_scores)

            # Get document stats
            stats = await self.vector_store_manager.get_collection_stats(
                collection_name
            )

            return {
                "summary": summary_response,
                "key_topics": key_topics,
                "document_count": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
            }

        except Exception as e:
            logger.error(f"Error generating chat summary: {e}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "key_topics": [],
                "document_count": 0,
            }

    async def _generate_summary_response(
        self, nodes_with_scores: List[NodeWithScore]
    ) -> str:
        """Generate summary response from nodes"""
        # Prepare context from nodes
        context_pieces = []
        for node_with_score in nodes_with_scores[:10]:  # Use top 10 chunks
            node = node_with_score.node
            source = node.metadata.get("file_name", "Unknown")
            context_pieces.append(f"From {source}: {node.text[:300]}...")

        context = "\n\n".join(context_pieces)

        system_prompt = (
            """You are an expert document summarizer. Create a comprehensive summary of the provided document excerpts.

Instructions:
1. Identify the main themes and topics across all documents
2. Highlight key information, findings, or conclusions
3. Organize the summary in a logical, easy-to-read format
4. Mention which documents contain which information
5. Keep the summary concise but informative

Document excerpts:
"""
            + context
        )

        user_prompt = "Please provide a comprehensive summary of these documents, highlighting the main topics and key information."

        try:
            response = await llm_client.generate_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.1,
            )
            return response
        except Exception as e:
            logger.error(f"Error generating summary response: {str(e)}")
            return f"Unable to generate summary due to technical issues: {str(e)}"

    def _extract_key_topics(self, nodes_with_scores: List[NodeWithScore]) -> List[str]:
        """Simple keyword extraction from retrieved nodes"""
        from collections import Counter
        import re

        all_text = " ".join(node.node.text for node in nodes_with_scores)

        # Extract potential keywords (simple approach)
        words = re.findall(r"\b[A-Z][a-z]+\b", all_text)  # Capitalized words

        # Count and return top keywords
        word_counts = Counter(words)
        key_topics = [word for word, count in word_counts.most_common(10) if count > 1]

        return key_topics[:5]  # Return top 5 topics
