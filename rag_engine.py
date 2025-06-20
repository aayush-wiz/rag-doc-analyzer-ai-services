import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Initialize ChromaDB client
# This points to the ChromaDB service running in Docker.
chroma_client = chromadb.HttpClient(host="oracyn_chroma_db", port=8000)

# Initialize Gemini models
llm = Gemini(
    model_name="models/gemini-1.5-flash-latest", api_key=os.getenv("GEMINI_API_KEY")
)
embed_model = GeminiEmbedding(
    model_name="models/text-embedding-004", api_key=os.getenv("GEMINI_API_KEY")
)


def process_document(document_path: str, chat_id: str):
    """Loads a document, creates an index, and persists it to ChromaDB."""
    try:
        # Create a ChromaVectorStore with a unique collection for the chat_id
        vector_store = ChromaVectorStore(
            chroma_collection=chroma_client.get_or_create_collection(chat_id)
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Load the document from the shared volume
        documents = SimpleDirectoryReader(input_files=[document_path]).load_data()

        # Create the index, which automatically handles chunking and embedding
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        print(f"Successfully processed and indexed document for chat_id: {chat_id}")
        return True
    except Exception as e:
        print(f"Error processing document: {e}")
        return False


def answer_query(query_text: str, chat_id: str):
    """Queries the index for a given chat_id to generate an answer."""
    try:
        # Load the index from the specific ChromaDB collection
        vector_store = ChromaVectorStore(
            chroma_collection=chroma_client.get_collection(chat_id)
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        # Create a query engine
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query(query_text)

        print(f"Successfully generated answer for chat_id: {chat_id}")
        return str(response)
    except Exception as e:
        print(f"Error answering query: {e}")
        return "An error occurred while generating the answer. The document may not be processed yet."
