import os
import base64
import tempfile
import json
from pathlib import Path
from dotenv import load_dotenv
import traceback
from typing import Optional, List, Dict

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

Settings.llm = Gemini(model_name="models/gemini-1.5-flash-latest", api_key=api_key)
Settings.embed_model = GeminiEmbedding(
    model_name="models/text-embedding-004", api_key=api_key
)
genai.configure(api_key=api_key)
print("LlamaIndex global settings configured.")

# --- Storage Path Configuration ---
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
VECTOR_STORAGE_PATH = Path(STORAGE_DIR) / "vector_storage"
VECTOR_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
print(f"Vector storage path set to: {VECTOR_STORAGE_PATH.resolve()}")


def get_persist_dir(chat_id: str) -> Path:
    return VECTOR_STORAGE_PATH / chat_id


# --- Core Logic Functions ---


def process_document_content(file_name: str, file_content_base64: str, chat_id: str):
    """
    Processes an uploaded document, creates a vector index using the default
    SimpleVectorStore, and persists it to disk.
    """
    persist_dir = get_persist_dir(chat_id)
    print(f"Processing document for chat_id: {chat_id}. Storage: {persist_dir}")

    try:
        file_content = base64.b64decode(file_content_base64)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / file_name
            temp_file_path.write_bytes(file_content)

            documents = SimpleDirectoryReader(input_files=[temp_file_path]).load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=persist_dir)

        print(f"Successfully processed and indexed content for chat_id: {chat_id}")
        return True
    except Exception:
        print(f"FATAL: Error processing document content for chat {chat_id}:")
        traceback.print_exc()
        return False


def answer_query(
    query_text: str, chat_id: str, history: Optional[List[Dict[str, str]]] = None
):
    """
    Answers a user query using the persisted index for the given chat.
    """
    persist_dir = get_persist_dir(chat_id)
    if not persist_dir.exists():
        return {
            "answer": f"Error: No document has been processed for this chat (ID: {chat_id}). Please upload a document first.",
            "tokens_used": 0,
        }

    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context)

        system_prompt = """
        You are Oracyn, a meticulous and expert AI research assistant. Your primary function is to provide detailed, accurate, and well-structured answers based *exclusively* on the document context provided.

        **Core Directives:**
        1.  **Grounding:** Your answers MUST be based **only** on the information contained within the provided documents. Do not use any external knowledge or make assumptions beyond what is written.
        2.  **Handling Missing Information:** If the answer to a question cannot be found in the provided context, you MUST state that clearly and concisely. For example, say "The provided document does not contain information about [the user's topic]." Do not attempt to guess the answer.
        3.  **No Apologies:** Do not apologize or mention that you are an AI. Be direct and confident in your responses.

        **Formatting Rules:**
        -   **Markdown:** Structure your entire response using Markdown for maximum clarity.
        -   **Structure:** Use headings (`## Sub-heading`), bold text (`**important term**`), and bulleted or numbered lists (`- ...` or `1. ...`) to break down complex information.
        -   **Tables:** If the user requests tabular data or if it's the best way to present comparisons, generate a clean Markdown table.
        -   **Code:** If you are presenting code, use Markdown code blocks with the correct language identifier (e.g., ```python ... ```).

        **Answering Style:**
        -   **Be Specific:** When quoting information or citing data from the document, be as specific as possible.
        -   **Comprehensive:** Provide thorough answers. Do not be overly brief unless the question calls for it.
        -   **Direct Start:** Begin your answer directly without introductory phrases like "Based on the provided context..." or "In the document...". The user already knows the context.
        """

        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        if history:
            for msg in history:
                memory.put(ChatMessage(role=msg["role"], content=msg["content"]))

        chat_engine = index.as_chat_engine(
            chat_mode="context", memory=memory, system_prompt=system_prompt
        )
        response = chat_engine.chat(query_text)

        total_tokens = 0
        metadata = getattr(response, "response_metadata", None) or getattr(
            response, "metadata", None
        )
        if metadata and isinstance(metadata, dict):
            token_usage = metadata.get("token_usage", {})
            total_tokens = token_usage.get("total_tokens", 0)
        else:
            print(
                f"WARN: Could not find token usage metadata in chat response for chat_id: {chat_id}."
            )

        print(
            f"Successfully generated answer for chat_id: {chat_id}, Tokens: {total_tokens}"
        )
        return {"answer": str(response), "tokens_used": total_tokens}
    except Exception:
        print(f"FATAL: Error answering query for chat {chat_id}:")
        traceback.print_exc()
        return {
            "answer": "An error occurred while generating the answer. The document index might be corrupted.",
            "tokens_used": 0,
        }


# Define a standard error response for when chart data cannot be found.
CHART_ERROR_JSON = json.dumps(
    {
        "error": "The requested data to generate a chart could not be found in the document. Please ask for a chart that can be built from the provided text."
    }
)
MAX_CONTEXT_CHARS_FOR_CHART = 16000


def generate_chart_data(prompt: str, chat_id: str, chart_type: str):
    """
    Generates structured JSON for a chart using the document context.
    """
    persist_dir = get_persist_dir(chat_id)
    if not persist_dir.exists():
        return None

    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context)

        doc_text = "\n".join(
            [doc.get_content() for doc in index.docstore.docs.values()]
        )
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        json_prompt = f"""
        You are a meticulous data visualization specialist. Your sole task is to analyze the provided document text and the user's request to generate a **perfectly formatted, raw JSON object** suitable for Chart.js.

        **CRITICAL GOAL:** The output JSON will be fed directly into a charting library without any intermediate processing. It must be 100% valid and adhere to the specified structure.

        **USER'S REQUEST:** "{prompt}"

        **DOCUMENT TEXT (analyzing first {MAX_CONTEXT_CHARS_FOR_CHART} characters):**
        ---
        {doc_text[:MAX_CONTEXT_CHARS_FOR_CHART]} 
        ---

        **RULESET (Follow Exactly):**

        1.  **JSON ONLY:** Your entire response MUST be the raw JSON object. Do NOT include any explanatory text, markdown ` ```json ... ``` ` wrappers, or any character outside of the JSON structure.

        2.  **DATA NOT FOUND:** If the document does NOT contain the specific quantifiable data needed to create the requested chart, you MUST NOT invent data. Instead, you MUST return the following exact JSON object:
            {CHART_ERROR_JSON}

        3.  **VALID DATA STRUCTURE:** If the data is found, the JSON object must have this precise structure:
            - A top-level `"type"` key (string, e.g., "{chart_type}").
            - A top-level `"data"` key (object).
            - The `"data"` object must contain:
                - `"labels"`: An array of strings.
                - `"datasets"`: An array of objects.
            - Each object inside `"datasets"` must contain:
                - `"label"`: A descriptive string for the legend.
                - `"data"`: An array of numbers.
                - **(Recommended)** `"backgroundColor"`: An array of visually appealing RGBA color strings (e.g., 'rgba(54, 162, 235, 0.6)').
                - **(Recommended)** `"borderColor"`: An array of corresponding solid RGB color strings (e.g., 'rgb(54, 162, 235)').

        **EXAMPLE OF PERFECT OUTPUT (for a 'bar' chart):**
        {{
          "type": "bar",
          "data": {{
            "labels": ["Q1 Sales", "Q2 Sales", "Q3 Sales", "Q4 Sales"],
            "datasets": [
              {{
                "label": "Revenue (USD)",
                "data": [120500, 195600, 301000, 500250],
                "backgroundColor": ["rgba(54, 162, 235, 0.6)"],
                "borderColor": ["rgb(54, 162, 235)"]
              }}
            ]
          }}
        }}

        Now, generate the raw JSON object based on the user's request and the document text.
        """

        response = model.generate_content(json_prompt)

        # --- THIS IS THE FIX ---
        # Manually count tokens for both input and output to ensure it always works.
        input_tokens = model.count_tokens(json_prompt).total_tokens
        output_tokens = model.count_tokens(response.text).total_tokens
        tokens_used = input_tokens + output_tokens
        # --- END OF THE FIX ---

        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        chart_json = json.loads(cleaned_text)

        if "error" in chart_json:
            print(f"Chart generation failed logically: {chart_json['error']}")
            return {"chart_json": chart_json, "tokens_used": tokens_used}

        print(f"Successfully generated chart JSON. Tokens used: {tokens_used}")
        return {"chart_json": chart_json, "tokens_used": tokens_used}
    except Exception as e:
        print(f"FATAL: Error generating chart data for chat {chat_id}:")
        traceback.print_exc()
        raise e
