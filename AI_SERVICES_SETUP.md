# RAG Document Analyzer - AI Services Setup Guide

## Overview

The RAG Document Analyzer now has a complete AI services backend built with FastAPI and modern LlamaIndex. This guide will help you set up and run both the backend and AI services.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │  AI Services    │
│   (React)       │◄──►│   (Express.js)  │◄──►│   (FastAPI)     │
│   Port: 5173    │    │   Port: 5001    │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                       ┌─────────────┐          ┌─────────────┐
                       │  Firebase   │          │  ChromaDB   │
                       │  Firestore  │          │ (Local DB)  │
                       │  Auth       │          │             │
                       └─────────────┘          └─────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │   Google    │
                       │ Cloud Store │
                       └─────────────┘
```

## Features

### Backend Features (Express.js)
- ✅ Firebase Authentication
- ✅ Google Cloud Storage for file storage
- ✅ Real-time updates with Socket.IO
- ✅ MVC architecture with controllers and routes
- ✅ File upload and management
- ✅ Query processing and history
- ✅ Analytics and monitoring

### AI Services Features (FastAPI)
- ✅ Modern LlamaIndex integration
- ✅ OpenAI GPT-4 and embeddings
- ✅ ChromaDB vector storage
- ✅ Document processing (PDF, DOCX, TXT, MD, XLSX, CSV)
- ✅ RAG query processing
- ✅ Automatic document vectorization
- ✅ Health monitoring endpoints
- ✅ Structured logging

## Setup Instructions

### 1. AI Services Setup

#### Prerequisites
- Python 3.8+
- OpenAI API key

#### Installation
```bash
cd ai-services

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp env.example .env
```

#### Configuration
Edit the `.env` file in `ai-services/`:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults are fine for development)
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
LOG_LEVEL=INFO

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE_MB=50

# RAG Configuration
SIMILARITY_TOP_K=5
RESPONSE_MODE=tree_summarize
```

#### Start AI Services
```bash
# Option 1: Using the startup script
python start.py

# Option 2: Using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Backend Setup

#### Prerequisites
- Node.js 16+
- Firebase project
- Google Cloud Storage bucket

#### Installation
```bash
cd backend

# Install dependencies
npm install

# Create environment file
cp env.example .env
```

#### Configuration
Edit the `.env` file in `backend/`:

```env
# Server
PORT=5001
NODE_ENV=development

# JWT
JWT_SECRET=your_super_secret_jwt_key_here
JWT_REFRESH_SECRET=your_refresh_secret_key_here

# Firebase (get from Firebase Console > Project Settings > Service Accounts)
FIREBASE_PROJECT_ID=your_firebase_project_id
FIREBASE_PRIVATE_KEY_ID=your_private_key_id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=your_service_account_email
FIREBASE_CLIENT_ID=your_client_id

# Google Cloud Storage
GCS_BUCKET_NAME=your_bucket_name
GCS_PROJECT_ID=your_gcs_project_id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json

# AI Services Integration
AI_SERVICE_URL=http://localhost:8000
AI_SERVICE_API_KEY=optional_api_key

# File Upload
MAX_FILE_SIZE=52428800  # 50MB
ALLOWED_FILE_TYPES=pdf,docx,doc,txt,csv,xlsx,md
```

#### Start Backend
```bash
npm start
# or for development with auto-reload
npm run dev
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## API Endpoints

### AI Services (FastAPI) - Port 8000

#### Health Check
- `GET /api/v1/health/` - Basic health check
- `GET /api/v1/health/detailed` - Detailed health with all services
- `GET /api/v1/health/vector-store` - Vector store specific health

#### Document Management
- `POST /api/v1/documents/upload` - Upload and process documents
- `DELETE /api/v1/documents/` - Delete documents by IDs
- `GET /api/v1/documents/info` - Get collection information
- `DELETE /api/v1/documents/reset` - Reset entire collection

#### Query Processing
- `POST /api/v1/query/` - Submit RAG query (structured)
- `POST /api/v1/query/simple` - Submit simple query
- `GET /api/v1/query/test` - Test query functionality

### Backend (Express.js) - Port 5001

#### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `POST /api/auth/refresh` - Refresh token
- `GET /api/auth/profile` - Get user profile

#### File Management
- `POST /api/files/upload` - Upload files (auto-processes through AI)
- `GET /api/files/` - Get user files
- `POST /api/files/reprocess` - Reprocess files
- `GET /api/files/:fileId/status` - Get file processing status
- `DELETE /api/files/:fileId` - Delete file

#### Query Processing
- `POST /api/queries/submit` - Submit query (forwards to AI services)
- `GET /api/queries/history` - Get query history
- `GET /api/queries/:queryId` - Get specific query
- `DELETE /api/queries/:queryId` - Delete query

#### Analytics
- `GET /api/analytics/dashboard` - Dashboard data
- `GET /api/analytics/usage` - Usage analytics
- `GET /api/analytics/queries` - Query analytics

## Document Processing Flow

1. **Upload**: User uploads files via frontend
2. **Storage**: Backend saves files to Google Cloud Storage
3. **Processing**: Backend automatically sends files to AI services
4. **Vectorization**: AI services process and embed documents into ChromaDB
5. **Completion**: Real-time updates notify frontend of completion
6. **Query**: Users can now query the processed documents

## Query Processing Flow

1. **Submit**: User submits query via frontend
2. **Forward**: Backend forwards query to AI services
3. **RAG**: AI services retrieve relevant documents and generate response
4. **Return**: Response with sources returned to user
5. **Store**: Query and response stored in Firestore for history

## Monitoring and Debugging

### AI Services Logs
```bash
cd ai-services
tail -f logs/app.log  # If logging to file
# or check console output
```

### Backend Logs
```bash
cd backend
npm run logs  # If configured
# or check console output
```

### Health Checks
- AI Services: `http://localhost:8000/api/v1/health/detailed`
- Backend: `http://localhost:5001/api/health`

## Testing

### Test AI Services
```bash
# Health check
curl http://localhost:8000/api/v1/health/

# Test query (requires documents to be uploaded first)
curl -X POST http://localhost:8000/api/v1/query/simple \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'
```

### Test Backend
```bash
# Health check
curl http://localhost:5001/api/health
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure virtual environment is activated and dependencies installed
2. **OpenAI API Errors**: Verify API key is correct and has credits
3. **ChromaDB Issues**: Delete `./chroma_db` folder to reset vector store
4. **File Upload Fails**: Check GCS credentials and bucket permissions
5. **Connection Refused**: Ensure AI services are running before starting backend

### Reset Everything
```bash
# Reset AI services vector store
curl -X DELETE http://localhost:8000/api/v1/documents/reset

# Or manually delete
cd ai-services
rm -rf chroma_db/
```

## Dependencies

### AI Services (Python)
- FastAPI 0.112.0
- LlamaIndex 0.11.20
- ChromaDB 0.5.23
- OpenAI (latest)
- Pydantic 2.8.2

### Backend (Node.js)
- Express.js
- Firebase Admin SDK
- Google Cloud Storage
- Socket.IO
- Axios

## Next Steps

1. **Production Deployment**: Configure for production environment
2. **Authentication**: Add API key authentication between services
3. **Rate Limiting**: Implement rate limiting for AI services
4. **Caching**: Add Redis caching for frequent queries
5. **Monitoring**: Add Prometheus metrics and monitoring
6. **Scaling**: Consider containerization with Docker

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify all environment variables are set correctly
3. Ensure all services are running and healthy
4. Check network connectivity between services 