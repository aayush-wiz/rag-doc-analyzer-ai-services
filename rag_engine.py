import os
import faiss
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage
# --- NEW IMPORTS ---
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
# ---
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from dotenv import load_dotenv

load_dotenv()

# --- FAISS Setup ---
FAISS_STORAGE_PATH = Path("/app/faiss_storage")
FAISS_STORAGE_PATH.mkdir(exist_ok=True)

# Initialize Gemini models
llm = Gemini(
    model_name="models/gemini-1.5-flash-latest", api_key=os.getenv("GEMINI_API_KEY")
)
embed_model = GeminiEmbedding(
    model_name="models/text-embedding-004", api_key=os.getenv("GEMINI_API_KEY")
)

def get_index_path(chat_id: str) -> str:
    """Helper function to create a unique file path for each chat's index."""
    return str(FAISS_STORAGE_PATH / f"{chat_id}.faiss")

def process_document(document_path: str, chat_id: str):
    """Loads a document, creates a FAISS index, and saves it to a file."""
    try:
        documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        index_path = get_index_path(chat_id)
        index.storage_context.persist(persist_dir=index_path)
        print(f"Successfully processed and saved FAISS index for chat_id: {chat_id}")
        return True
    except Exception as e:
        print(f"Error processing document: {e}")
        return False

def answer_query(query_text: str, chat_id: str, history: list = None):
    """Loads a FAISS index and uses a ChatEngine to generate a conversational answer."""
    try:
        index_path = get_index_path(chat_id)
        
        if not os.path.exists(index_path):
             return "Document not found. Please upload the document for this chat first."

        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        
        # --- NEW CHAT ENGINE LOGIC ---
        # Create a memory buffer to hold the chat history
        # We limit the memory to the last 10 messages to keep it efficient
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        
        # If history is provided, load it into the memory buffer
        if history:
            for msg in history:
                # Convert our simple dict message into the LlamaIndex ChatMessage format
                memory.put(ChatMessage(role=msg['role'], content=msg['content']))

        # Create the ChatEngine, configured to use our index and memory
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=(
                "You are Oracyn, a helpful AI assistant. Answer the user's questions based "
                "on the context provided from their documents. Be friendly and precise."
            ),
        )

        # Stream the response from the chat engine
        response = chat_engine.chat(query_text)
        # --- END OF NEW LOGIC ---

        print(f"Successfully generated conversational answer for chat_id: {chat_id}")
        return str(response)
    except Exception as e:
        print(f"Error answering query: {e}")
        return "An error occurred while generating the answer."