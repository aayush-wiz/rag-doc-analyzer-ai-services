FROM python:3.11-slim

# Install system dependencies needed by your Python packages
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-root user and group to run the application
# This is a critical security best practice.
RUN addgroup --system pythonapp && adduser --system --group pythonapp

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Change the ownership of the application code to the new non-root user
RUN chown -R pythonapp:pythonapp /app

# Switch to the non-root user
USER pythonapp

# Expose the port the Uvicorn server will run on
EXPOSE 8000

# This is the production start command.
# It runs Uvicorn without the --reload flag for efficiency and stability.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]