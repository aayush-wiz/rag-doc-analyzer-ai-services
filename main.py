import grpc
from concurrent import futures
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

# Import the generated gRPC files and the LlamaIndex logic
import rag_pb2
import rag_pb2_grpc
from rag_engine import process_document, answer_query

load_dotenv()


# --- gRPC Service Implementation ---
class RagServicer(rag_pb2_grpc.RagServiceServicer):
    def ProcessDocument(self, request, context):
        print(f"gRPC: Received ProcessDocument request for chat_id: {request.chat_id}")
        success = process_document(request.document_path, request.chat_id)
        if success:
            return rag_pb2.ProcessResponse(
                success=True, message="Document processed successfully."
            )
        else:
            return rag_pb2.ProcessResponse(
                success=False, message="Failed to process document."
            )

    def AnswerQuery(self, request, context):
        print(f"gRPC: Received AnswerQuery request for chat_id: {request.chat_id}")
        answer = answer_query(request.query_text, request.chat_id)
        return rag_pb2.QueryResponse(answer=answer)


# --- FastAPI App ---
# We use FastAPI for a health check endpoint, though it's not strictly necessary.
# The main purpose is that uvicorn, which runs FastAPI, can also manage our gRPC server.
app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "AI Service is running"}


# --- Server Startup ---
def serve():
    # Create a gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rag_pb2_grpc.add_RagServiceServicer_to_server(RagServicer(), server)

    # Listen on port 50051 for gRPC requests
    print("Starting gRPC server on port 50051...")
    server.add_insecure_port("[::]:50051")
    server.start()

    # Start the FastAPI app using uvicorn
    # This keeps the script running and serves the health check endpoint.
    print("Starting FastAPI health check on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # This part is just to keep the server alive
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
