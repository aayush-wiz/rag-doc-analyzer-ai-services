import os
import base64
import tempfile
import json
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader, StorageContext, VectorStoreIndex,
    load_index_from_storage, Settings
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import google.generativeai as genai

load_dotenv()

# --- Global Settings Configuration ---
print("Configuring LlamaIndex global settings...")
Settings.llm = Gemini(model_name="models/gemini-1.5-flash-latest", api_key=os.getenv("GEMINI_API_KEY"))
Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=os.getenv("GEMINI_API_KEY"))
print("LlamaIndex global settings configured.")

# --- FAISS Setup ---
FAISS_STORAGE_PATH = Path("/app/faiss_storage")
FAISS_STORAGE_PATH.mkdir(exist_ok=True)


def get_index_path(chat_id: str) -> str:
    return str(FAISS_STORAGE_PATH / f"{chat_id}.faiss")

def process_document_content(file_name: str, file_content_base64: str, chat_id: str):
    try:
        file_content = base64.b64decode(file_content_base64)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / file_name
            with open(temp_file_path, "wb") as f:
                f.write(file_content)
            
            documents = SimpleDirectoryReader(input_files=[temp_file_path]).load_data()
            index = VectorStoreIndex.from_documents(documents)
            index_path = get_index_path(chat_id)
            index.storage_context.persist(persist_dir=index_path)
        print(f"Successfully processed content for chat_id: {chat_id}")
        return True
    except Exception as e:
        print(f"Error processing document content: {e}")
        return False

def answer_query(query_text: str, chat_id: str, history: list = None):
    index_path = get_index_path(chat_id)
    try:
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        
        if history:
            for msg in history:
                memory.put(ChatMessage(role=msg.role, content=msg.content))

        chat_engine = index.as_chat_engine(
            chat_mode="context", memory=memory,
            system_prompt="You are Oracyn, a helpful AI assistant. Answer based on the documents."
        )
        response = chat_engine.chat(query_text)
        print(f"Successfully generated conversational answer for chat_id: {chat_id}")
        return str(response)
    except FileNotFoundError:
        return "Document not found. Please upload the document for this chat first."
    except Exception as e:
        print(f"Error answering query: {e}")
        return "An error occurred while generating the answer."

# @desc    Generate structured JSON data for a chart.
def generate_chart_data(prompt: str, chat_id: str, chart_type: str):
    index_path = get_index_path(chat_id)
    try:
        # THIS IS THE KEY FIX: Load the index from storage first.
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        
        # Now, safely access the full text from the document store.
        doc_text = "\n".join([doc.get_content() for doc in index.docstore.docs.values()])

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        json_prompt = f"""
        Act as a data analysis expert. Analyze the following document text and the user's request
        to generate a valid JSON object suitable for a {chart_type} chart using a library like Chart.js or Recharts.

        USER'S REQUEST: "{prompt}"

        DOCUMENT TEXT:
        ---
        {doc_text[:8000]} 
        ---

        Your response MUST be ONLY the raw JSON object and nothing else. Do not wrap it in ```json ... ``` or any other text.
        The JSON object must have two top-level keys: "type" and "data".
        The "data" object must contain "labels" (an array of strings) and "datasets" (an array of objects).
        Each object in "datasets" must have a "label" (a string) and "data" (an array of numbers).

        Now, generate the raw JSON object for the user's request.
        """
        
        response = model.generate_content(json_prompt)
        
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        chart_json = json.loads(cleaned_text)
        
        print("Successfully generated chart JSON data.")
        return chart_json

    except Exception as e:
        print(f"Error generating chart data: {e}")
        return None