# =================================================================
# FILE: oracyn-ai-service/rag_engine.py (DEFINITIVE FIX)
# This version fixes the ImportError by renaming the function to
# `process_document_content` to match what main.py is calling.
# =================================================================
import os
import base64
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

load_dotenv()

# --- Global Settings Configuration ---
print("Configuring LlamaIndex global settings...")
Settings.llm = Gemini(
    model_name="models/gemini-1.5-flash-latest", api_key=os.getenv("GEMINI_API_KEY")
)
Settings.embed_model = GeminiEmbedding(
    model_name="models/text-embedding-004", api_key=os.getenv("GEMINI_API_KEY")
)
print("LlamaIndex global settings configured.")

# --- FAISS Setup ---
FAISS_STORAGE_PATH = Path("/app/faiss_storage")
FAISS_STORAGE_PATH.mkdir(exist_ok=True)

def get_index_path(chat_id: str) -> str:
    """Helper function to create a unique file path for each chat's index."""
    return str(FAISS_STORAGE_PATH / f"{chat_id}.faiss")

# THIS IS THE KEY FIX: The function is now correctly named `process_document_content`
def process_document_content(file_name: str, file_content_base64: str, chat_id: str):
    """Decodes Base64 content, saves to a temp file, and processes it."""
    try:
        file_content = base64.b64decode(file_content_base64)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / file_name
            with open(temp_file_path, "wb") as f:
                f.write(file_content)
            
            print(f"Temporary file created at: {temp_file_path}")

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
    """Loads a FAISS index and uses a ChatEngine to generate a conversational answer."""
    index_path = get_index_path(chat_id)
    
    try:
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        
        if history:
            for msg in history:
                memory.put(ChatMessage(role=msg.role, content=msg.content))

        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt="You are Oracyn, a helpful AI assistant. Answer based on the documents.",
        )

        response = chat_engine.chat(query_text)
        print(f"Successfully generated conversational answer for chat_id: {chat_id}")
        return str(response)

    except FileNotFoundError:
        print(f"Index not found for chat_id: {chat_id}. Prompting user to upload.")
        return "Document not found. Please upload the document for this chat first."
    except Exception as e:
        print(f"Error answering query: {e}")
        return "An error occurred while generating the answer."

