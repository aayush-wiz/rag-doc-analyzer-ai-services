import os
import base64
import tempfile
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
from llama_index.core import (
    SimpleDirectoryReader,  # type: ignore
    StorageContext,  # type: ignore
    VectorStoreIndex,  # type: ignore
    load_index_from_storage,  # type: ignore
    Settings,  # type: ignore
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage  # type: ignore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding  # type: ignore
import google.generativeai as genai

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


def answer_query(query_text: str, chat_id: str, history: Optional[list] = None):
    index_path = get_index_path(chat_id)
    try:
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        if history:
            for msg in history:
                memory.put(ChatMessage(role=msg.role, content=msg.content))

            system_prompt = (
                """
        You are Oracyn, a helpful and precise AI assistant. Your primary function is to answer user queries based on the provided document context.

        Follow these rules strictly:
        1.  **Format your entire response using Markdown.** Use headings, lists, bold text, and code blocks where appropriate to structure your answer clearly.
        2.  If the user asks for data in a table, generate a Markdown table.
        3.  If you are providing code snippets, use Markdown code blocks with the correct language identifier (e.g., ```python ... ```).
        4.  Base your answers *only* on the context from the documents provided. If the information is not in the documents, state that clearly. Do not use external knowledge.
        """,
            )
        chat_engine = index.as_chat_engine(
            chat_mode="context", memory=memory, system_prompt=system_prompt
        )

        response = chat_engine.chat(query_text)

        # Extract token usage from LlamaIndex response metadata
        token_usage = response.metadata.get("token_usage", {})
        total_tokens = token_usage.get("total_tokens", 0)

        print(
            f"Successfully generated answer for chat_id: {chat_id}, Tokens: {total_tokens}"
        )
        # Return both the answer and the token count
        return {"answer": str(response), "tokens_used": total_tokens}
    except FileNotFoundError:
        return {"answer": "Document not found...", "tokens_used": 0}
    except Exception as e:
        print(f"Error answering query: {e}")
        return {
            "answer": "An error occurred while generating the answer.",
            "tokens_used": 0,
        }


def generate_chart_data(prompt: str, chat_id: str, chart_type: str):
    index_path = get_index_path(chat_id)
    try:
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        doc_text = "\n".join(
            [doc.get_content() for doc in index.docstore.docs.values()]
        )

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        json_prompt = f"""
        Act as an expert data analyst and a frontend developer's assistant. Your task is to analyze the following document text and the user's request to generate a valid JSON object perfectly formatted for Chart.js.

        USER'S REQUEST: "{prompt}"

        DOCUMENT TEXT (first 12000 characters):
        ---
        {doc_text[:12000]}
        ---

        **CRITICAL INSTRUCTIONS:**
        1.  Your response MUST BE **ONLY** the raw JSON object. Do not include any explanatory text, markdown, or code block wrappers like ```json ... ```. The entire output must be parsable with `JSON.parse()`.
        2.  The JSON object must have a `type` key (e.g., "{chart_type}") and a `data` key.
        3.  The `data` object **MUST** contain a `labels` key (an array of strings) and a `datasets` key (an array of objects).
        4.  Each object inside the `datasets` array **MUST** have a `label` (a string for the legend) and `data` (an array of numbers corresponding to the labels).
        5.  For better visuals, you can optionally include `backgroundColor` (an array of RGBA strings like 'rgba(54, 162, 235, 0.5)') and `borderColor` (an array of RGB strings like 'rgb(54, 162, 235)') in the dataset objects. The length of these color arrays should match the length of the data array.

        EXAMPLE OF A PERFECT OUTPUT FOR A BAR CHART:
        {{
          "type": "bar",
          "data": {{
            "labels": ["Q1", "Q2", "Q3", "Q4"],
            "datasets": [
              {{
                "label": "Sales 2024",
                "data": [120, 190, 300, 500],
                "backgroundColor": ["rgba(54, 162, 235, 0.5)"],
                "borderColor": ["rgb(54, 162, 235)"]
              }}
            ]
          }}
        }}

        Now, generate the raw JSON object based on the user's request and the document text.
        """
        response = model.generate_content(json_prompt)

        # Extract token usage from the Gemini API response
        tokens_used = model.count_tokens(json_prompt + response.text).total_tokens

        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        chart_json = json.loads(cleaned_text)

        print(f"Successfully generated chart JSON. Tokens used: {tokens_used}")
        # Return both the chart data and the token count
        return {"chart_json": chart_json, "tokens_used": tokens_used}

    except Exception as e:
        print(f"Error generating chart data: {e}")
        return None
