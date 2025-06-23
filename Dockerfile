FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgomp1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# THIS IS THE KEY CHANGE FOR THE AI SERVICE:
# The --reload flag tells uvicorn to watch for file changes.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]