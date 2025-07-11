from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from dotenv import load_dotenv
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Import core logic functions
from rag_engine import process_document_content, answer_query, generate_chart_data

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Oracyn AI Service",
    description="A service for processing documents and answering queries using a RAG pipeline with ChromaDB.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
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
    history: Optional[List[ChatMessage]] = None


class ChartRequest(BaseModel):
    prompt: str
    chat_id: str
    chart_type: str


# --- API Endpoints ---
@app.get("/", summary="Health Check")
def health_check():
    """Confirms service is running."""
    return {"status": "ORACYN AI Service is running."}


@app.post("/process-document", summary="Process and Index a Document")
async def process_document_endpoint(request: DocumentRequest):
    """Processes a document and creates a vector index with ChromaDB."""
    print(f"REST: Process-document request for chat_id: {request.chat_id}")

    result = process_document_content(
        file_name=request.file_name,
        file_content_base64=request.file_content_base64,
        chat_id=request.chat_id,
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "message": f"Document '{request.file_name}' processed successfully for chat {request.chat_id}.",
        "success": True,
    }


@app.post("/answer-query", summary="Answer a Query")
async def answer_query_endpoint(request: QueryRequest):
    """Receives a query and returns an AI-generated answer."""
    print(f"REST: Answer-query request for chat_id: {request.chat_id}")

    history_dicts: Optional[List[Dict[str, str]]] = None
    if request.history:
        history_dicts = [msg.model_dump() for msg in request.history]

    response_data = answer_query(
        query_text=request.query_text,
        chat_id=request.chat_id,
        history=history_dicts,
    )

    if not response_data["success"]:
        raise HTTPException(status_code=500, detail=response_data["answer"])

    return response_data


@app.post("/generate-chart", summary="Generate Chart Data")
async def generate_chart_endpoint(request: ChartRequest):
    """Generates Chart.js-compatible JSON data."""
    print(f"REST: Generate-chart request for chat_id: {request.chat_id}")

    response_data = generate_chart_data(
        prompt=request.prompt, chat_id=request.chat_id, chart_type=request.chart_type
    )

    if not response_data["success"]:
        raise HTTPException(
            status_code=400, detail=response_data["chart_json"]["error"]
        )

    return response_data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
