import os
import base64
import tempfile
import json
from pathlib import Path
from dotenv import load_dotenv
import traceback
from typing import Optional, List, Dict, Tuple
import mimetypes
import hashlib
import chromadb
from chromadb.config import Settings as ChromaSettings

# --- Core LlamaIndex Imports ---
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.vector_stores.chroma import ChromaVectorStore

# --- LlamaIndex Integration Imports ---
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# --- Third-Party Imports ---
import google.generativeai as genai

# --- Load Environment & Global Config ---
load_dotenv()
print("Environment variables loaded.")

print("Configuring LlamaIndex global settings...")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable not set! Please check your .env file."
    )

Settings.llm = Gemini(model_name="models/gemini-1.5-pro", api_key=api_key)
Settings.embed_model = GeminiEmbedding(
    model_name="models/text-embedding-004", api_key=api_key
)
genai.configure(api_key=api_key)
print("LlamaIndex global settings configured.")

# --- Storage Path Configuration ---
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
VECTOR_STORAGE_PATH = Path(STORAGE_DIR) / "vector_storage"
HISTORY_STORAGE_PATH = Path(STORAGE_DIR) / "chat_history"
VECTOR_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
HISTORY_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
print(f"Vector storage path set to: {VECTOR_STORAGE_PATH.resolve()}")
print(f"History storage path set to: {HISTORY_STORAGE_PATH.resolve()}")

# --- ChromaDB Configuration ---
chroma_client = chromadb.PersistentClient(
    path=str(VECTOR_STORAGE_PATH), settings=ChromaSettings(anonymized_telemetry=False)
)

# --- Supported File Types ---
SUPPORTED_FILE_TYPES = {
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
}


def get_persist_dir(chat_id: str) -> Path:
    return VECTOR_STORAGE_PATH / chat_id


def get_history_file(chat_id: str) -> Path:
    return HISTORY_STORAGE_PATH / f"{chat_id}_history.json"


def validate_file(file_name: str, file_content: bytes) -> Tuple[bool, str]:
    """Validates file type and size."""
    file_extension = Path(file_name).suffix.lower()
    if file_extension not in SUPPORTED_FILE_TYPES:
        return False, f"Unsupported file type: {file_extension}"

    mime_type, _ = mimetypes.guess_type(file_name)
    if mime_type not in SUPPORTED_FILE_TYPES.values():
        return False, f"Invalid MIME type for {file_name}"

    if len(file_content) > 20 * 1024 * 1024:  # 20MB limit
        return False, "File size exceeds 20MB limit"

    return True, ""


def save_chat_history(chat_id: str, history: List[Dict[str, str]]) -> bool:
    """Persists chat history to disk."""
    try:
        history_file = get_history_file(chat_id)
        with history_file.open("w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved chat history for chat_id: {chat_id}")
        return True
    except Exception as e:
        print(f"Error saving chat history for chat_id {chat_id}: {str(e)}")
        return False


def load_chat_history(chat_id: str) -> List[Dict[str, str]]:
    """Loads chat history from disk."""
    history_file = get_history_file(chat_id)
    if not history_file.exists():
        return []

    try:
        with history_file.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading chat history for chat_id {chat_id}: {str(e)}")
        return []


def process_document_content(file_name: str, file_content_base64: str, chat_id: str):
    """
    Processes an uploaded document, creates a vector index using ChromaDB, and persists it.
    """
    persist_dir = get_persist_dir(chat_id)
    print(f"Processing document for chat_id: {chat_id}. Storage: {persist_dir}")

    try:
        file_content = base64.b64decode(file_content_base64)

        # Validate file
        is_valid, error_msg = validate_file(file_name, file_content)
        if not is_valid:
            print(f"Validation failed: {error_msg}")
            return {"success": False, "error": error_msg}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / file_name
            temp_file_path.write_bytes(file_content)

            documents = SimpleDirectoryReader(input_files=[temp_file_path]).load_data()

            # Generate document fingerprint
            doc_hash = hashlib.sha256(file_content).hexdigest()

            # Initialize ChromaDB vector store
            chroma_collection = chroma_client.get_or_create_collection(
                f"chat_{chat_id}"
            )
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Create and persist index
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            index.storage_context.persist(persist_dir=persist_dir)

            # Store document metadata
            metadata_file = persist_dir / "metadata.json"
            metadata = {
                "file_name": file_name,
                "doc_hash": doc_hash,
                "processed_at": str(Path(temp_file_path).stat().st_mtime),
            }
            with metadata_file.open("w") as f:
                json.dump(metadata, f)

        print(f"Successfully processed and indexed content for chat_id: {chat_id}")
        return {"success": True, "error": None}
    except Exception:
        print(f"FATAL: Error processing document content for chat {chat_id}:")
        traceback.print_exc()
        return {"success": False, "error": "Failed to process document"}


def answer_query(
    query_text: str, chat_id: str, history: Optional[List[Dict[str, str]]] = None
):
    """
    Answers a user query using the persisted ChromaDB index and maintains chat history.
    """
    persist_dir = get_persist_dir(chat_id)
    if not persist_dir.exists():
        return {
            "answer": f"No document processed for chat_id: {chat_id}. Please upload a document first.",
            "tokens_used": 0,
            "success": False,
        }

    try:
        # Load ChromaDB vector store
        chroma_collection = chroma_client.get_or_create_collection(f"chat_{chat_id}")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=str(persist_dir)
        )
        index = load_index_from_storage(storage_context)

        system_prompt = """
        You are Oracyn, an expert AI research assistant. Answer queries based solely on the provided document context.

        **Core Directives:**
        - Ground answers in document content only.
        - If information is missing, state: "The document does not contain information about [topic]."
        - Use Markdown for clear formatting with headings, lists, and tables.
        - Provide specific, comprehensive, and direct answers.
        - For complex queries, break down answers into sections.
        - Use examples or quotes from the document when relevant.
        """

        # Load and merge history
        full_history = load_chat_history(chat_id)
        if history:
            full_history.extend(history)

        memory = ChatMemoryBuffer.from_defaults(token_limit=5000)
        for msg in full_history:
            memory.put(ChatMessage(role=msg["role"], content=msg["content"]))

        memory.put(ChatMessage(role="user", content=query_text))

        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=system_prompt,
            similarity_top_k=5,
        )
        response = chat_engine.chat(query_text)

        total_tokens = 0
        metadata = getattr(response, "response_metadata", None) or getattr(
            response, "metadata", None
        )
        if metadata and isinstance(metadata, dict):
            token_usage = metadata.get("token_usage", {})
            total_tokens = token_usage.get("total_tokens", 0)

        # Update chat history
        full_history.append({"role": "user", "content": query_text})
        full_history.append({"role": "assistant", "content": str(response)})
        save_chat_history(chat_id, full_history)

        print(f"Generated answer for chat_id: {chat_id}, Tokens: {total_tokens}")
        return {"answer": str(response), "tokens_used": total_tokens, "success": True}
    except Exception:
        print(f"FATAL: Error answering query for chat {chat_id}:")
        traceback.print_exc()
        return {
            "answer": "An error occurred while generating the answer.",
            "tokens_used": 0,
            "success": False,
        }


CHART_ERROR_JSON = json.dumps(
    {
        "error": "The requested data to generate a chart could not be found in the document."
    }
)
MAX_CONTEXT_CHARS_FOR_CHART = 50000


def generate_chart_data(prompt: str, chat_id: str, chart_type: str):
    """
    Generates structured JSON for a chart using the document context.
    """
    persist_dir = get_persist_dir(chat_id)
    if not persist_dir.exists():
        return {"chart_json": CHART_ERROR_JSON, "tokens_used": 0, "success": False}

    try:
        # Load ChromaDB vector store
        chroma_collection = chroma_client.get_or_create_collection(f"chat_{chat_id}")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=str(persist_dir)
        )
        index = load_index_from_storage(storage_context)

        doc_text = "\n".join(
            [doc.get_content() for doc in index.docstore.docs.values()]
        )
        model = genai.GenerativeModel("gemini-1.5-pro")

        json_prompt = f"""
        You are an expert data visualization specialist. Generate a raw JSON object for Chart.js based on the user's request and document text.

        **USER'S REQUEST:** "{prompt}"
        **CHART TYPE:** "{chart_type}"
        **DOCUMENT TEXT (first {MAX_CONTEXT_CHARS_FOR_CHART} chars):**
        ---
        {doc_text[:MAX_CONTEXT_CHARS_FOR_CHART]}
        ---

        **RULES:**
        1. Output raw JSON only, no wrappers or explanations.
        2. If data is missing, return: {CHART_ERROR_JSON}
        3. Structure:
           - "type": string (e.g., "{chart_type}")
           - "data":
             - "labels": string array
             - "datasets": array of objects with "label", "data", "backgroundColor", "borderColor"
           - "options": object with sensible defaults for the chart type
        4. Ensure data is quantifiable and matches the chart type.
        5. Use visually distinct colors for datasets.
        6. For numerical data, ensure accuracy and consistency with document content.
        7. For time-series data, format dates appropriately.
        8. Validate that the chart type is supported by Chart.js (e.g., bar, line, pie).
        """

        response = model.generate_content(json_prompt)
        input_tokens = model.count_tokens(json_prompt).total_tokens
        output_tokens = model.count_tokens(response.text).total_tokens
        tokens_used = input_tokens + output_tokens

        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        chart_json = json.loads(cleaned_text)

        if "error" in chart_json:
            print(f"Chart generation failed: {chart_json['error']}")
            return {
                "chart_json": chart_json,
                "tokens_used": tokens_used,
                "success": False,
            }

        print(f"Generated chart JSON for chat_id: {chat_id}. Tokens: {tokens_used}")
        return {"chart_json": chart_json, "tokens_used": tokens_used, "success": True}
    except Exception:
        print(f"FATAL: Error generating chart data for chat {chat_id}:")
        traceback.print_exc()
        return {"chart_json": CHART_ERROR_JSON, "tokens_used": 0, "success": False}
