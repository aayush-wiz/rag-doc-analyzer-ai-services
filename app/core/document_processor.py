# ai-services/app/core/document_processor.py
import logging
import asyncio
import tempfile
import hashlib
import uuid
from typing import Dict, Any, List
from pathlib import Path

import aiohttp
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import (
    PDFReader,
    DocxReader,
    FlatReader,
    CSVReader,
    MarkdownReader,
)

from app.config.settings import settings
from app.core.llm_client import llm_client
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Complete document processing pipeline with LlamaIndex integration"""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager

        # Initialize document readers
        self.readers = {
            ".pdf": PDFReader(),
            ".docx": DocxReader(),
            ".doc": DocxReader(),
            ".txt": FlatReader(),
            ".md": MarkdownReader(),
            ".csv": CSVReader(),
        }

        # Initialize text splitter with your settings
        self.text_splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    async def process_document(
        self,
        chat_id: int,
        document_id: int,
        file_path: str,
        file_name: str,
        file_type: str,
    ) -> Dict[str, Any]:
        """Process document and store embeddings"""

        logger.info(f"Processing document {document_id}: {file_name}")

        try:
            # Step 1: Download file from R2
            local_file_path = await self._download_file(file_path, file_name)

            try:
                # Step 2: Extract text using LlamaIndex
                documents = await self._extract_text(local_file_path, file_name)

                # Step 3: Split into chunks
                nodes = await self._split_text(
                    documents, document_id, chat_id, file_name, file_type
                )

                # Step 4: Generate embeddings
                nodes_with_embeddings = await self._generate_embeddings(nodes)

                # Step 5: Store in vector database
                await self._store_embeddings(nodes_with_embeddings, chat_id)

                # Calculate stats
                total_characters = sum(len(node.text) for node in nodes)
                avg_node_size = total_characters / len(nodes) if nodes else 0

                return {
                    "success": True,
                    "message": f"Document {file_name} processed successfully",
                    "stats": {
                        "total_documents": 1,
                        "total_nodes": len(nodes),
                        "total_characters": total_characters,
                        "average_node_size": round(avg_node_size, 2),
                    },
                }

            finally:
                # Always clean up temp file
                Path(local_file_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to process document {file_name}",
                "error": str(e),
            }

    async def _download_file(self, file_path: str, file_name: str) -> str:
        """Download file from R2 storage"""
        try:
            # Create temporary file
            temp_dir = Path(tempfile.gettempdir())
            local_file_path = temp_dir / f"{uuid.uuid4()}_{file_name}"

            # For development, handle both URL and local file scenarios
            if file_path.startswith(("http://", "https://")):
                # Download from URL
                async with aiohttp.ClientSession() as session:
                    async with session.get(file_path) as response:
                        if response.status == 200:
                            content = await response.read()
                            with open(local_file_path, "wb") as f:
                                f.write(content)
                        else:
                            raise Exception(
                                f"Failed to download file: HTTP {response.status}"
                            )
            else:
                # Handle local file path (for testing)
                if Path(file_path).exists():
                    import shutil

                    shutil.copy2(file_path, local_file_path)
                else:
                    raise FileNotFoundError(f"File not found: {file_path}")

            logger.info(f"Downloaded file to {local_file_path}")
            return str(local_file_path)

        except Exception as e:
            logger.error(f"Error downloading file {file_path}: {str(e)}")
            raise

    async def _extract_text(self, file_path: str, file_name: str) -> List:
        """Extract text using LlamaIndex readers"""
        try:
            file_extension = Path(file_name).suffix.lower()

            if file_extension not in self.readers:
                raise ValueError(f"Unsupported file type: {file_extension}")

            reader = self.readers[file_extension]

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(None, reader.load_data, file_path)

            logger.info(f"Extracted {len(documents)} documents from {file_name}")
            return documents

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise

    async def _split_text(
        self,
        documents: List,
        document_id: int,
        chat_id: int,
        file_name: str,
        file_type: str,
    ) -> List[TextNode]:
        """Split text into chunks using LlamaIndex"""
        try:
            all_nodes = []

            for doc_idx, document in enumerate(documents):
                # Split document into nodes
                nodes = self.text_splitter.get_nodes_from_documents([document])

                # Add metadata to each node
                for node_idx, node in enumerate(nodes):
                    # Ensure node has proper ID
                    if not hasattr(node, "node_id") or not node.node_id:
                        node.node_id = str(uuid.uuid4())

                    # Add comprehensive metadata
                    node.metadata.update(
                        {
                            "document_id": document_id,
                            "chat_id": chat_id,
                            "file_name": file_name,
                            "file_type": file_type,
                            "doc_idx": doc_idx,
                            "node_idx": node_idx,
                            "chunk_id": f"{document_id}_{doc_idx}_{node_idx}",
                        }
                    )

                all_nodes.extend(nodes)

            logger.info(f"Split into {len(all_nodes)} nodes")
            return all_nodes

        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            raise

    async def _generate_embeddings(self, nodes: List[TextNode]) -> List[TextNode]:
        """Generate embeddings for nodes"""
        try:
            logger.info(f"Generating embeddings for {len(nodes)} nodes...")

            # Process nodes in batches to avoid overwhelming the API
            batch_size = 5
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i : i + batch_size]

                for node in batch:
                    try:
                        # Generate embedding for node text
                        embedding = await llm_client.generate_embedding(node.text)
                        node.embedding = embedding
                    except Exception as e:
                        logger.error(
                            f"Error generating embedding for node {node.node_id}: {str(e)}"
                        )
                        # Use zero embedding as fallback
                        node.embedding = [0.0] * 384  # Standard dimension

                # Small delay between batches to respect rate limits
                await asyncio.sleep(0.1)

            logger.info(f"Generated embeddings for {len(nodes)} nodes")
            return nodes

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def _store_embeddings(self, nodes: List[TextNode], chat_id: int):
        """Store embeddings in vector database"""
        try:
            collection_name = f"chat_{chat_id}"
            await self.vector_store_manager.add_nodes(nodes, collection_name)
            logger.info(f"Stored {len(nodes)} nodes in collection {collection_name}")

        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            raise

    async def delete_document(self, chat_id: int, document_id: int) -> bool:
        """Delete document from vector store"""
        try:
            collection_name = f"chat_{chat_id}"
            success = await self.vector_store_manager.delete_document(
                collection_name, document_id
            )

            if success:
                logger.info(f"Deleted document {document_id} from chat {chat_id}")
            else:
                logger.warning(
                    f"Failed to delete document {document_id} from chat {chat_id}"
                )

            return success

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False

    async def get_document_stats(self, chat_id: int) -> Dict[str, Any]:
        """Get document statistics for a chat"""
        try:
            collection_name = f"chat_{chat_id}"
            stats = await self.vector_store_manager.get_collection_stats(
                collection_name
            )
            return stats

        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "collection_name": f"chat_{chat_id}",
                "error": str(e),
            }

    def generate_document_hash(self, content: bytes) -> str:
        """Generate hash for document deduplication"""
        return hashlib.sha256(content).hexdigest()

    async def check_document_exists(self, chat_id: int, document_hash: str) -> bool:
        """Check if document with same hash already exists"""
        try:
            collection_name = f"chat_{chat_id}"
            return await self.vector_store_manager.document_exists(
                collection_name, document_hash
            )
        except Exception:
            return False
