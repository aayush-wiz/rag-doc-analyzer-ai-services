import os
import faiss
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from dotenv import load_dotenv

load_dotenv()

# --- FAISS Setup ---
# Define the path where FAISS index files will be stored inside the Docker volume
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
        # Load the document
        documents = SimpleDirectoryReader(input_files=[document_path]).load_data()

        # Create the FAISS index in memory
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        
        # Save the index to a file
        index_path = get_index_path(chat_id)
        index.storage_context.persist(persist_dir=index_path)

        print(f"Successfully processed and saved FAISS index for chat_id: {chat_id}")
        return True
    except Exception as e:
        print(f"Error processing document: {e}")
        return False


def answer_query(query_text: str, chat_id: str):
    """Loads a FAISS index from a file and queries it to generate an answer."""
    try:
        index_path = get_index_path(chat_id)
        
        if not os.path.exists(index_path):
             return "Document not found. Please upload the document for this chat first."

        # Load the index from the file
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = VectorStoreIndex.from_storage(storage_context, embed_model=embed_model)

        # Create a query engine
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query(query_text)

        print(f"Successfully generated answer for chat_id: {chat_id}")
        return str(response)
    except Exception as e:
        print(f"Error answering query: {e}")
        return "An error occurred while generating the answer."
