# =================================================================
# FILE: oracyn-ai-service/Dockerfile (DEFINITIVE FIX)
# This version has a simplified command that no longer runs the
# problematic bootstrap script.
# =================================================================

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's source code
COPY . .

# Final CMD: Directly start the FastAPI/gRPC server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]