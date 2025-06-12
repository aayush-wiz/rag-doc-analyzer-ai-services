# ai-service/app/core/document_processor.py (UPDATED for Gemini/Claude)
import logging
import tempfile
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.file import (
    PDFReader,
    DocxReader,
    CSVReader,
    UnstructuredReader,
)

from app.config.settings import settings
from app.utils.file_utils import download_file_from_r2, get_file_extension
from app.core.vector_store import VectorStoreManager
from app.core.llm_client import llm_client

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Document processing service using LlamaIndex with Gemini/Claude"""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager

        # Document readers
        self.readers = {
            ".pdf": PDFReader(),
            ".docx": DocxReader(),
            ".doc": DocxReader(),
            ".csv": CSVReader(),
            ".txt": UnstructuredReader(),
            ".xlsx": UnstructuredReader(),
            ".xls": UnstructuredReader(),
            ".pptx": UnstructuredReader(),
            ".ppt": UnstructuredReader(),
        }

        # Setup ingestion pipeline with custom embedding
        self.text_splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
        )

    async def process_document(
        self,
        chat_id: int,
        document_id: int,
        file_path: str,
        file_name: str,
        file_type: str,
    ) -> Dict[str, Any]:
        """
        Process a document and store embeddings
        """
        logger.info(f"Processing document {document_id}: {file_name}")

        try:
            # Download file from R2 storage
            file_content = await download_file_from_r2(file_path)

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix=get_file_extension(file_name), delete=False
            ) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            try:
                # Load and parse document
                documents = await self._load_document(
                    temp_file_path, file_name, file_type
                )

                # Add metadata to documents
                for doc in documents:
                    doc.metadata.update(
                        {
                            "chat_id": chat_id,
                            "document_id": document_id,
                            "file_name": file_name,
                            "file_type": file_type,
                            "file_path": file_path,
                        }
                    )

                # Process documents into nodes
                nodes = await self._process_documents_to_nodes(documents)

                # Generate embeddings for nodes
                await self._generate_embeddings_for_nodes(nodes)

                # Store in vector database
                collection_name = f"chat_{chat_id}"
                await self.vector_store_manager.add_nodes(nodes, collection_name)

                # Calculate processing stats
                stats = {
                    "total_documents": len(documents),
                    "total_nodes": len(nodes),
                    "total_characters": sum(len(doc.text) for doc in documents),
                    "average_node_size": (
                        sum(len(node.text) for node in nodes) / len(nodes)
                        if nodes
                        else 0
                    ),
                }

                logger.info(f"Successfully processed document {document_id}: {stats}")

                return {
                    "success": True,
                    "document_id": document_id,
                    "stats": stats,
                    "message": f"Processed {len(documents)} documents into {len(nodes)} chunks",
                }

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            return {
                "success": False,
                "document_id": document_id,
                "error": str(e),
                "message": f"Failed to process document: {str(e)}",
            }

    async def _load_document(
        self, file_path: str, file_name: str, file_type: str
    ) -> List[Document]:
        """Load document using appropriate reader"""

        file_extension = get_file_extension(file_name)

        if file_extension not in self.readers:
            raise ValueError(f"Unsupported file type: {file_extension}")

        reader = self.readers[file_extension]

        try:
            # Load documents
            documents = reader.load_data(file=Path(file_path))

            # Add base metadata
            for doc in documents:
                doc.metadata.update(
                    {
                        "source_file": file_name,
                        "file_type": file_type,
                        "file_extension": file_extension,
                    }
                )

            logger.info(f"Loaded {len(documents)} documents from {file_name}")
            return documents

        except Exception as e:
            logger.error(f"Error loading document {file_name}: {str(e)}")
            raise ValueError(f"Failed to load document: {str(e)}")

    async def _process_documents_to_nodes(self, documents: List[Document]):
        """Process documents into text nodes"""
        from llama_index.core.schema import TextNode
        import uuid

        all_nodes = []

        for doc in documents:
            # Split document into chunks
            text_chunks = self.text_splitter.split_text(doc.text)

            # Create nodes from chunks
            for i, chunk in enumerate(text_chunks):
                node = TextNode(
                    text=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "node_id": str(uuid.uuid4()),
                    },
                )
                node.node_id = node.metadata["node_id"]
                all_nodes.append(node)

        return all_nodes

    async def _generate_embeddings_for_nodes(self, nodes):
        """Generate embeddings for all nodes"""
        logger.info(f"Generating embeddings for {len(nodes)} nodes...")

        # Extract texts
        texts = [node.text for node in nodes]

        # Generate embeddings in batches
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_nodes = nodes[i : i + batch_size]

            try:
                embeddings = await llm_client.generate_embeddings_batch(batch_texts)

                # Assign embeddings to nodes
                for node, embedding in zip(batch_nodes, embeddings):
                    node.embedding = embedding

            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {str(e)}")
                # Generate embeddings individually as fallback
                for node in batch_nodes:
                    try:
                        embedding = await llm_client.generate_embedding(node.text)
                        node.embedding = embedding
                    except Exception as node_error:
                        logger.error(
                            f"Failed to generate embedding for node: {str(node_error)}"
                        )
                        # Use zero embedding as fallback
                        node.embedding = [
                            0.0
                        ] * 384  # Default dimension for all-MiniLM-L6-v2

        logger.info("Embeddings generation completed")

    async def delete_document(self, chat_id: int, document_id: int) -> bool:
        """Delete document embeddings from vector store"""
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
        """Get statistics for all documents in a chat"""
        try:
            collection_name = f"chat_{chat_id}"
            stats = await self.vector_store_manager.get_collection_stats(
                collection_name
            )
            return stats

        except Exception as e:
            logger.error(f"Error getting document stats for chat {chat_id}: {str(e)}")
            return {"error": str(e), "total_documents": 0, "total_chunks": 0}

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
