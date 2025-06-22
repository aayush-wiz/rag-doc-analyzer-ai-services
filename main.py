from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from rag_engine import process_document, answer_query

load_dotenv()

# THIS LINE IS THE KEY FIX for the AI service error.
app = FastAPI()

# --- Pydantic Models for Request Bodies ---
class DocumentRequest(BaseModel):
    document_path: str
    chat_id: str

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query_text: str
    chat_id: str
    history: Optional[List[ChatMessage]] = None

# --- API Endpoints ---
@app.get("/")
def health_check():
    return {"status": "ORACYN AI Service is running"}

@app.post("/process-document")
async def process_document_endpoint(request: DocumentRequest):
    print(f"REST: Received process-document request for chat_id: {request.chat_id}")
    success = process_document(request.document_path, request.chat_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process document")
    return {"message": "Document processing initiated successfully."}

@app.post("/answer-query")
async def answer_query_endpoint(request: QueryRequest):
    print(f"REST: Received answer-query request for chat_id: {request.chat_id}")
    answer = answer_query(request.query_text, request.chat_id, request.history)
    return {"answer": answer}