# FILE: oracyn-ai-service/Dockerfile

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

# Command to run the application
# We use uvicorn to run the FastAPI app, which will also start our gRPC server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]