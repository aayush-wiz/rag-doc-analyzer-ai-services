import os
import tempfile
from typing import List, Optional, Dict, Any
import asyncio
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import PDFReader, DocxReader

from app.core.config import settings
from app.core.logging import logger


class VectorStoreService:
    """Service for managing vector store operations with LlamaIndex and ChromaDB."""
    
    def __init__(self):
        self.chroma_client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self.vector_store: Optional[ChromaVectorStore] = None
        self.storage_context: Optional[StorageContext] = None
        self.index: Optional[VectorStoreIndex] = None
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the vector store service."""
        try:
            logger.info("Initializing VectorStoreService...")
            
            # Configure LlamaIndex settings
            Settings.llm = OpenAI(
                model=settings.OPENAI_MODEL,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                api_key=settings.OPENAI_API_KEY
            )
            
            Settings.embed_model = OpenAIEmbedding(
                model=settings.OPENAI_EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY
            )
            
            Settings.chunk_size = settings.CHUNK_SIZE
            Settings.chunk_overlap = settings.CHUNK_OVERLAP
            
            # Initialize ChromaDB
            await self._initialize_chroma()
            
            # Initialize index
            await self._initialize_index()
            
            self.initialized = True
            logger.info("VectorStoreService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorStoreService: {e}")
            raise
    
    async def _initialize_chroma(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME
            )
            
            # Create vector store
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            
            # Create storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            logger.info(f"ChromaDB initialized with collection: {settings.CHROMA_COLLECTION_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def _initialize_index(self) -> None:
        """Initialize or load the vector index."""
        try:
            # Check if collection has documents
            collection_count = self.collection.count()
            
            if collection_count > 0:
                # Load existing index
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    storage_context=self.storage_context
                )
                logger.info(f"Loaded existing index with {collection_count} documents")
            else:
                # Create empty index
                self.index = VectorStoreIndex(
                    nodes=[],
                    storage_context=self.storage_context
                )
                logger.info("Created new empty index")
                
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            raise
    
    async def add_documents(
        self,
        file_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add documents to the vector store."""
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info(f"Adding {len(file_paths)} documents to vector store")
            
            # Load documents
            documents = []
            for file_path in file_paths:
                file_docs = await self._load_document(file_path, metadata)
                documents.extend(file_docs)
            
            if not documents:
                return {"success": False, "message": "No documents to add"}
            
            # Create nodes with text splitter
            text_splitter = SentenceSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            
            nodes = text_splitter.get_nodes_from_documents(documents)
            
            # Add to index
            self.index.insert_nodes(nodes)
            
            result = {
                "success": True,
                "documents_added": len(documents),
                "nodes_created": len(nodes),
                "collection_count": self.collection.count()
            }
            
            logger.info(f"Successfully added documents: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return {"success": False, "error": str(e)}
    
    async def _load_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Load a single document from file path."""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == ".pdf":
                reader = PDFReader()
            elif file_extension in [".docx", ".doc"]:
                reader = DocxReader()
            else:
                # Use SimpleDirectoryReader for other formats
                reader = SimpleDirectoryReader(input_files=[file_path])
                return reader.load_data()
            
            # Load with specific reader
            documents = reader.load_data(file=Path(file_path))
            
            # Add metadata
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
                    doc.metadata["file_path"] = file_path
                    doc.metadata["file_name"] = Path(file_path).name
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            return []
    
    async def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query the vector store."""
        if not self.initialized:
            await self.initialize()
        
        try:
            top_k = top_k or settings.SIMILARITY_TOP_K
            
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                response_mode=settings.RESPONSE_MODE
            )
            
            # Execute query
            response = query_engine.query(query_text)
            
            # Extract source information
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    sources.append({
                        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "metadata": node.metadata,
                        "score": getattr(node, 'score', None)
                    })
            
            result = {
                "response": str(response),
                "sources": sources,
                "query": query_text,
                "top_k": top_k
            }
            
            logger.info(f"Query executed successfully: {query_text[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise
    
    async def delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """Delete documents from the vector store."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # ChromaDB delete by IDs
            self.collection.delete(ids=document_ids)
            
            # Recreate index
            await self._initialize_index()
            
            result = {
                "success": True,
                "deleted_count": len(document_ids),
                "remaining_count": self.collection.count()
            }
            
            logger.info(f"Deleted {len(document_ids)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self.initialized:
            await self.initialize()
        
        try:
            count = self.collection.count()
            
            return {
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "document_count": count,
                "persist_directory": settings.CHROMA_PERSIST_DIRECTORY
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector store service."""
        try:
            if not self.initialized:
                return {"status": "not_initialized"}
            
            # Try to get collection count
            count = self.collection.count()
            
            return {
                "status": "healthy",
                "document_count": count,
                "chroma_db": "connected"
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def reset_collection(self) -> Dict[str, Any]:
        """Reset the entire collection (delete all documents)."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Delete the collection
            self.chroma_client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
            
            # Recreate collection and index
            await self._initialize_chroma()
            await self._initialize_index()
            
            logger.info("Collection reset successfully")
            return {"success": True, "message": "Collection reset successfully"}
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return {"success": False, "error": str(e)}


# Create global service instance
vector_store_service = VectorStoreService() 