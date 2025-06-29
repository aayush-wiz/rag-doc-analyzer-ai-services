from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from dotenv import load_dotenv

# Import the new function from the rag_engine
from rag_engine import process_document_content, answer_query, generate_chart_data

load_dotenv()
app = FastAPI()


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


# NEW model for chart requests
class ChartRequest(BaseModel):
    prompt: str  # e.g., "Create a bar chart of my sales data by quarter"
    chat_id: str
    chart_type: str  # e.g., "bar", "line", "pie"


# --- API Endpoints ---
@app.get("/")
def health_check():
    return {"status": "ORACYN AI Service is running"}


@app.post("/process-document")
async def process_document_endpoint(request: DocumentRequest):
    print(
        f"REST: Received process-document content request for chat_id: {request.chat_id}"
    )
    success = process_document_content(
        request.file_name, request.file_content_base64, request.chat_id
    )
    if not success:
        raise HTTPException(
            status_code=500, detail="Failed to process document content"
        )
    return {"message": "Document content processed successfully."}


@app.post("/answer-query")
async def answer_query_endpoint(request: QueryRequest):
    print(f"REST: Received answer-query request for chat_id: {request.chat_id}")
    # The function now returns a dictionary
    response_data = answer_query(request.query_text, request.chat_id, request.history)
    return response_data


@app.post("/generate-chart")
async def generate_chart_endpoint(request: ChartRequest):
    print(f"REST: Received generate-chart request for chat_id: {request.chat_id}")
    # The function now returns a dictionary
    response_data = generate_chart_data(
        request.prompt, request.chat_id, request.chart_type
    )
    if not response_data:
        raise HTTPException(status_code=500, detail="Failed to generate chart data")
    # Return the entire dictionary
    return response_data
