#!/usr/bin/env python3
"""
Simple ChromaDB Web Viewer
A basic Flask web interface to view ChromaDB data
Usage: python scripts/chromadb_web_viewer.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from flask import Flask, render_template_string, request, jsonify
    import chromadb
    from chromadb.config import Settings
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install flask chromadb python-dotenv")
    sys.exit(1)

# Load environment
load_dotenv()

app = Flask(__name__)

# Initialize ChromaDB
chroma_path = Path("chroma_db")
if not chroma_path.exists():
    print(f"❌ ChromaDB directory not found: {chroma_path}")
    sys.exit(1)

try:
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False, allow_reset=False),
    )
    print(f"✅ Connected to ChromaDB at: {chroma_path.absolute()}")
except Exception as e:
    print(f"❌ Failed to connect to ChromaDB: {e}")
    sys.exit(1)

# HTML Templates
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ChromaDB Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { background: #4CAF50; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .collection { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; background: #fafafa; }
        .document { border-left: 4px solid #4CAF50; padding: 10px; margin: 10px 0; background: white; }
        .metadata { background: #e8f5e8; padding: 8px; margin: 5px 0; border-radius: 3px; font-size: 0.9em; }
        .content { margin: 10px 0; line-height: 1.5; }
        .search-box { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
        .btn { background: #4CAF50; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #45a049; }
        .stats { display: flex; gap: 20px; margin: 15px 0; }
        .stat-box { background: #e3f2fd; padding: 10px; border-radius: 5px; text-align: center; flex: 1; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗃️ ChromaDB Database Viewer</h1>
            <p>Explore your vector database collections and documents</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>{{ stats.total_collections }}</h3>
                <p>Collections</p>
            </div>
            <div class="stat-box">
                <h3>{{ stats.total_documents }}</h3>
                <p>Documents</p>
            </div>
            <div class="stat-box">
                <h3>{{ stats.database_size }}</h3>
                <p>Database Size</p>
            </div>
        </div>

        {% for collection in collections %}
        <div class="collection">
            <h3>📚 Collection: {{ collection.name }}</h3>
            <p><strong>Documents:</strong> {{ collection.count }}</p>
            {% if collection.metadata %}
            <div class="metadata">
                <strong>Metadata:</strong> {{ collection.metadata }}
            </div>
            {% endif %}
            
            <form method="POST" action="/search">
                <input type="hidden" name="collection_name" value="{{ collection.name }}">
                <input type="text" name="query" placeholder="Search in {{ collection.name }}..." class="search-box">
                <button type="submit" class="btn">🔍 Search</button>
            </form>
            
            <a href="/collection/{{ collection.name }}" class="btn">📖 View Documents</a>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

COLLECTION_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ChromaDB - {{ collection_name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { background: #4CAF50; color: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .document { border-left: 4px solid #4CAF50; padding: 15px; margin: 15px 0; background: #fafafa; border-radius: 4px; }
        .metadata { background: #e8f5e8; padding: 8px; margin: 5px 0; border-radius: 3px; font-size: 0.9em; }
        .content { margin: 10px 0; line-height: 1.5; white-space: pre-wrap; }
        .back-btn { background: #2196F3; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; display: inline-block; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Collections</a>
        
        <div class="header">
            <h1>📚 Collection: {{ collection_name }}</h1>
            <p>{{ document_count }} documents</p>
        </div>

        {% for doc in documents %}
        <div class="document">
            <h4>📄 Document ID: {{ doc.id }}</h4>
            {% if doc.metadata %}
            <div class="metadata">
                <strong>Metadata:</strong> {{ doc.metadata }}
            </div>
            {% endif %}
            {% if doc.content %}
            <div class="content">{{ doc.content }}</div>
            {% endif %}
            {% if doc.embedding_dim %}
            <p><small>🧮 Embedding: {{ doc.embedding_dim }} dimensions</small></p>
            {% endif %}
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""


@app.route("/")
def index():
    """Main page showing all collections"""
    try:
        collections = client.list_collections()
        collection_data = []
        total_documents = 0

        for collection in collections:
            count = collection.count()
            total_documents += count
            collection_data.append(
                {
                    "name": collection.name,
                    "count": count,
                    "metadata": collection.metadata,
                }
            )

        # Calculate database size
        size_bytes = sum(
            f.stat().st_size for f in chroma_path.rglob("*") if f.is_file()
        )
        size_mb = size_bytes / (1024 * 1024)

        stats = {
            "total_collections": len(collections),
            "total_documents": total_documents,
            "database_size": f"{size_mb:.1f} MB",
        }

        return render_template_string(
            MAIN_TEMPLATE, collections=collection_data, stats=stats
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", 500


@app.route("/collection/<collection_name>")
def view_collection(collection_name):
    """View documents in a specific collection"""
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()

        # Get documents (limit to first 50 for performance)
        results = collection.get(
            limit=min(50, count), include=["documents", "metadatas", "embeddings"]
        )

        documents = []
        for i, doc_id in enumerate(results["ids"]):
            doc_data = {
                "id": doc_id,
                "content": results["documents"][i] if results["documents"] else None,
                "metadata": results["metadatas"][i] if results["metadatas"] else None,
                "embedding_dim": (
                    len(results["embeddings"][i])
                    if results["embeddings"] and results["embeddings"][i]
                    else None
                ),
            }
            documents.append(doc_data)

        return render_template_string(
            COLLECTION_TEMPLATE,
            collection_name=collection_name,
            document_count=count,
            documents=documents,
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", 500


@app.route("/search", methods=["POST"])
def search():
    """Search documents in a collection"""
    try:
        collection_name = request.form["collection_name"]
        query = request.form["query"]

        collection = client.get_collection(collection_name)

        # Search documents
        results = collection.query(
            query_texts=[query],
            n_results=10,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        if results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc_data = {
                    "id": doc_id,
                    "content": (
                        results["documents"][0][i] if results["documents"][0] else None
                    ),
                    "metadata": (
                        results["metadatas"][0][i] if results["metadatas"][0] else None
                    ),
                    "distance": (
                        results["distances"][0][i] if results["distances"][0] else None
                    ),
                }
                documents.append(doc_data)

        # Modify template to show search results
        search_template = COLLECTION_TEMPLATE.replace(
            "<h1>📚 Collection: {{ collection_name }}</h1>",
            '<h1>🔍 Search Results in: {{ collection_name }}</h1><p>Query: "'
            + query
            + '"</p>',
        )

        return render_template_string(
            search_template,
            collection_name=collection_name,
            document_count=len(documents),
            documents=documents,
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", 500


if __name__ == "__main__":
    print("🌐 Starting ChromaDB Web Viewer...")
    print("📱 Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5001)
