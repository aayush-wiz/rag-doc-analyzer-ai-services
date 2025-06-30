"""
main.py: Defines the FastAPI application and its API endpoints.
This file handles incoming HTTP requests, validates them using Pydantic,
and calls the appropriate logic functions from rag_engine.py
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from dotenv import load_dotenv

# Import the core logic functions from our RAG engine
from rag_engine import process_document_content, answer_query, generate_chart_data

# Load environment variables from a .env file
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(
    title="Oracyn AI Service",
    description="A service for processing documents and answering queries using a RAG pipeline.",
    version="1.0.0",
)


# --- Pydantic Models for Request Validation ---


class DocumentRequest(BaseModel):
    file_name: str
    file_content_base64: str
    chat_id: str


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query_text: str
    chat_id: str
    # The history is optional and will be a list of ChatMessage objects
    history: Optional[List[ChatMessage]] = None


class ChartRequest(BaseModel):
    prompt: str
    chat_id: str
    chart_type: str


# --- API Endpoints ---


@app.get("/", summary="Health Check")
def health_check():
    """A simple health check endpoint to confirm the service is running."""
    return {"status": "ORACYN AI Service is running and ready."}


@app.post("/process-document", summary="Process and Index a Document")
async def process_document_endpoint(request: DocumentRequest):
    """
    Receives a document, processes it, and creates a vector index.
    This is the first step for any new chat.
    """
    print(f"REST: Received process-document request for chat_id: {request.chat_id}")

    success = process_document_content(
        file_name=request.file_name,
        file_content_base64=request.file_content_base64,
        chat_id=request.chat_id,
    )

    if not success:
        raise HTTPException(
            status_code=500, detail="Failed to process document content in RAG engine."
        )

    return {
        "message": f"Document '{request.file_name}' processed successfully for chat {request.chat_id}."
    }


@app.post("/answer-query", summary="Answer a Query")
async def answer_query_endpoint(request: QueryRequest):
    """
    Receives a user's query and chat history, and returns an AI-generated answer.
    """
    print(f"REST: Received answer-query request for chat_id: {request.chat_id}")

    # --- START OF THE FIX ---
    # Convert the list of Pydantic ChatMessage models into a list of simple dictionaries,
    # which is what the `answer_query` function expects.
    history_dicts: Optional[List[Dict[str, str]]] = None
    if request.history:
        history_dicts = [msg.model_dump() for msg in request.history]
    # --- END OF THE FIX ---

    response_data = answer_query(
        query_text=request.query_text,
        chat_id=request.chat_id,
        history=history_dicts,  # Pass the corrected list of dicts
    )

    return response_data


@app.post("/generate-chart", summary="Generate Chart Data")
async def generate_chart_endpoint(request: ChartRequest):
    """
    Receives a prompt to generate a chart and returns Chart.js-compatible JSON data.
    """
    print(f"REST: Received generate-chart request for chat_id: {request.chat_id}")

    response_data = generate_chart_data(
        prompt=request.prompt, chat_id=request.chat_id, chart_type=request.chart_type
    )

    if not response_data:
        raise HTTPException(
            status_code=500, detail="Failed to generate chart data in RAG engine."
        )

    return response_data
