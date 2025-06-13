# ai-services/app/core/vector_store.py (WORKING VERSION)
import logging
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core.schema import NodeWithScore, BaseNode

from app.config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages ChromaDB vector store operations"""

    def __init__(self):
        self.client = None
        self.collections = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def initialize(self):
        """Initialize ChromaDB client"""
        try:
            # Initialize ChromaDB client based on environment
            if settings.ENVIRONMENT == "development":
                # Use persistent local storage for development
                self.client = chromadb.PersistentClient(
                    path=settings.CHROMA_PERSIST_DIRECTORY,
                    settings=ChromaSettings(
                        allow_reset=False, anonymized_telemetry=False
                    ),
                )
                logger.info(
                    f"✅ ChromaDB initialized (Local): {settings.CHROMA_PERSIST_DIRECTORY}"
                )
            else:
                # Use HTTP client for production
                self.client = chromadb.HttpClient(
                    host=settings.CHROMA_HOST,
                    port=settings.CHROMA_PORT,
                    settings=ChromaSettings(
                        allow_reset=False, anonymized_telemetry=False
                    ),
                )
                logger.info(
                    f"✅ ChromaDB initialized (Remote): {settings.CHROMA_HOST}:{settings.CHROMA_PORT}"
                )

            # Test connection
            await self._test_connection()
            logger.info("✅ ChromaDB client initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB client: {e}")
            raise

    async def _test_connection(self):
        """Test ChromaDB connection"""

        def _test():
            # Simple test to verify connection
            collections = self.client.list_collections()
            return len(collections) >= 0

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, _test)

    async def get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection"""
        if collection_name in self.collections:
            return self.collections[collection_name]

        def _get_or_create():
            try:
                # Try to get existing collection
                collection = self.client.get_collection(collection_name)
                logger.info(f"Retrieved existing collection: {collection_name}")
            except Exception:
                # Create new collection if it doesn't exist
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "description": f"Document embeddings for {collection_name}"
                    },
                )
                logger.info(f"Created new collection: {collection_name}")

            return collection

        loop = asyncio.get_event_loop()
        collection = await loop.run_in_executor(self.executor, _get_or_create)

        self.collections[collection_name] = collection
        return collection

    async def add_nodes(self, nodes: List[BaseNode], collection_name: str):
        """Add nodes to vector store"""
        if not nodes:
            logger.warning("No nodes to add to vector store")
            return

        try:
            collection = await self.get_or_create_collection(collection_name)

            def _add_nodes():
                # Prepare data for ChromaDB
                documents = []
                embeddings = []
                metadatas = []
                ids = []

                for node in nodes:
                    documents.append(node.text)
                    embeddings.append(node.embedding)

                    # Prepare metadata (ChromaDB requires string values)
                    metadata = {
                        "document_id": str(node.metadata.get("document_id", "")),
                        "chat_id": str(node.metadata.get("chat_id", "")),
                        "file_name": str(node.metadata.get("file_name", "")),
                        "file_type": str(node.metadata.get("file_type", "")),
                        "node_id": str(node.node_id),
                    }
                    metadatas.append(metadata)
                    ids.append(node.node_id)

                # Add to collection
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )

                return len(nodes)

            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(self.executor, _add_nodes)

            logger.info(f"Added {count} nodes to collection {collection_name}")

        except Exception as e:
            logger.error(f"Error adding nodes to vector store: {e}")
            raise

    async def search_similar(
        self,
        query_embedding: List[float],
        collection_name: str,
        n_results: int = None,
        where_filter: Dict = None,
    ) -> List[NodeWithScore]:
        """Search for similar documents"""
        try:
            collection = await self.get_or_create_collection(collection_name)
            n_results = n_results or settings.MAX_SEARCH_RESULTS

            def _search():
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )
                return results

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(self.executor, _search)

            # Convert to NodeWithScore objects
            nodes_with_scores = []

            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ):
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    score = max(0, 1 - distance)

                    # Only include results above similarity threshold
                    if score >= settings.SIMILARITY_THRESHOLD:
                        node = BaseNode(
                            text=doc,
                            metadata=metadata,
                            node_id=metadata.get("node_id", f"node_{i}"),
                        )

                        node_with_score = NodeWithScore(node=node, score=score)
                        nodes_with_scores.append(node_with_score)

            logger.info(
                f"Found {len(nodes_with_scores)} similar documents in {collection_name}"
            )
            return nodes_with_scores

        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []

    async def delete_document(self, collection_name: str, document_id: int) -> bool:
        """Delete all nodes for a specific document"""
        try:
            collection = await self.get_or_create_collection(collection_name)

            def _delete():
                # Delete by document_id metadata
                collection.delete(where={"document_id": str(document_id)})
                return True

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, _delete)

            logger.info(f"Deleted document {document_id} from {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete entire collection"""
        try:

            def _delete():
                self.client.delete_collection(collection_name)
                return True

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, _delete)

            # Remove from cache
            if collection_name in self.collections:
                del self.collections[collection_name]

            logger.info(f"Deleted collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        try:
            collection = await self.get_or_create_collection(collection_name)

            def _get_stats():
                count = collection.count()

                # Get unique documents count
                unique_docs = collection.get(include=["metadatas"])
                doc_ids = set()
                if unique_docs["metadatas"]:
                    doc_ids = {
                        meta.get("document_id") for meta in unique_docs["metadatas"]
                    }

                return {
                    "total_chunks": count,
                    "total_documents": len(doc_ids),
                    "collection_name": collection_name,
                }

            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(self.executor, _get_stats)

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "collection_name": collection_name,
                "error": str(e),
            }

    async def document_exists(self, collection_name: str, document_hash: str) -> bool:
        """Check if document with hash exists"""
        try:
            collection = await self.get_or_create_collection(collection_name)

            def _check():
                results = collection.get(
                    where={"document_hash": document_hash}, limit=1
                )
                return len(results["ids"]) > 0

            loop = asyncio.get_event_loop()
            exists = await loop.run_in_executor(self.executor, _check)

            return exists

        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return False

    async def close(self):
        """Close connections and cleanup"""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.collections.clear()
        logger.info("Vector store connections closed")
