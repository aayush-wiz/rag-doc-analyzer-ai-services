from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from app.services.vector_store import vector_store_service
from app.core.logging import logger

router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for query operations."""
    query: str = Field(..., description="The query text", min_length=1)
    top_k: Optional[int] = Field(None, description="Number of top results to return", ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters to apply")


class QueryResponse(BaseModel):
    """Response model for query operations."""
    response: str = Field(..., description="The generated response")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents used")
    query: str = Field(..., description="The original query")
    top_k: int = Field(..., description="Number of results returned")


@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Query documents using RAG (Retrieval-Augmented Generation)."""
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Execute query
        result = await vector_store_service.query(
            query_text=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        return QueryResponse(
            response=result["response"],
            sources=result["sources"],
            query=result["query"],
            top_k=result["top_k"]
        )
        
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simple")
async def simple_query(query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    """Simple query endpoint that accepts query as a string parameter."""
    try:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Processing simple query: {query[:100]}...")
        
        # Execute query
        result = await vector_store_service.query(
            query_text=query.strip(),
            top_k=top_k
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process simple query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test")
async def test_query() -> Dict[str, Any]:
    """Test endpoint to verify query functionality."""
    try:
        test_query = "What is this document about?"
        
        logger.info("Running test query...")
        
        # Get collection info first
        collection_info = await vector_store_service.get_collection_info()
        
        if collection_info.get("document_count", 0) == 0:
            return {
                "status": "no_documents",
                "message": "No documents available for querying",
                "collection_info": collection_info
            }
        
        # Execute test query
        result = await vector_store_service.query(
            query_text=test_query,
            top_k=3
        )
        
        return {
            "status": "success",
            "test_query": test_query,
            "result": result,
            "collection_info": collection_info
        }
        
    except Exception as e:
        logger.error(f"Test query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 