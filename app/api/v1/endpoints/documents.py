from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import tempfile
import os
from pathlib import Path

from app.services.vector_store import vector_store_service
from app.core.config import settings
from app.core.logging import logger

router = APIRouter()


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """Upload and process documents."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate file types and sizes
        for file in files:
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in settings.SUPPORTED_FILE_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"File type {file_extension} not supported. Supported types: {settings.SUPPORTED_FILE_TYPES}"
                )
        
        # Save files temporarily and process
        temp_files = []
        try:
            for file in files:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(file.filename).suffix,
                    dir=settings.UPLOAD_DIRECTORY
                )
                
                # Write file content
                content = await file.read()
                temp_file.write(content)
                temp_file.close()
                
                temp_files.append(temp_file.name)
                logger.info(f"Saved temporary file: {temp_file.name}")
            
            # Process metadata
            doc_metadata = {}
            if metadata:
                try:
                    import json
                    doc_metadata = json.loads(metadata)
                except Exception:
                    logger.warning("Failed to parse metadata, ignoring")
            
            # Add documents to vector store
            result = await vector_store_service.add_documents(
                file_paths=temp_files,
                metadata=doc_metadata
            )
            
            if result.get("success"):
                return {
                    "success": True,
                    "message": "Documents uploaded and processed successfully",
                    "files_processed": len(files),
                    "documents_added": result.get("documents_added", 0),
                    "nodes_created": result.get("nodes_created", 0)
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process documents: {result.get('error', 'Unknown error')}"
                )
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file}: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/")
async def delete_documents(document_ids: List[str]) -> Dict[str, Any]:
    """Delete documents by IDs."""
    try:
        if not document_ids:
            raise HTTPException(status_code=400, detail="No document IDs provided")
        
        result = await vector_store_service.delete_documents(document_ids)
        
        if result.get("success"):
            return {
                "success": True,
                "message": f"Deleted {result.get('deleted_count', 0)} documents",
                "deleted_count": result.get("deleted_count", 0),
                "remaining_count": result.get("remaining_count", 0)
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete documents: {result.get('error', 'Unknown error')}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_collection_info() -> Dict[str, Any]:
    """Get information about the document collection."""
    try:
        return await vector_store_service.get_collection_info()
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reset")
async def reset_collection() -> Dict[str, Any]:
    """Reset the entire document collection (delete all documents)."""
    try:
        result = await vector_store_service.reset_collection()
        
        if result.get("success"):
            return {
                "success": True,
                "message": "Collection reset successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to reset collection: {result.get('error', 'Unknown error')}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset collection: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 