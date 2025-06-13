#!/usr/bin/env python3
"""
Test Document Processing Script
Creates a test document and processes it through your RAG pipeline
"""

import asyncio
import aiohttp
import json
import tempfile
from pathlib import Path


async def create_test_document():
    """Create a test document for processing"""
    test_content = """# Test Document for RAG System

## Introduction
This is a comprehensive test document designed to verify the RAG (Retrieval Augmented Generation) system functionality.

## Key Information
- **Project**: DocAnalyzer RAG System
- **Technology Stack**: FastAPI, LlamaIndex, ChromaDB, Gemini/Claude
- **Purpose**: Document analysis and question answering

## Important Facts
1. The system can process multiple file formats including PDF, Word, and text files
2. Documents are split into chunks for optimal retrieval
3. Vector embeddings are generated for semantic search
4. The system supports real-time question answering

## Business Benefits
- Faster information retrieval
- Improved decision making
- Automated document analysis
- Cost-effective knowledge management

## Technical Details
The RAG system uses:
- **Vector Database**: ChromaDB for storing embeddings
- **LLM Integration**: Gemini as primary, Claude as fallback
- **Document Processing**: LlamaIndex for text extraction and chunking
- **Embedding Model**: Sentence transformers for semantic understanding

## Conclusion
This test document contains various types of information that should be retrievable through the RAG system, including technical details, business benefits, and structured data.
"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(test_content)
        return f.name


async def test_document_processing():
    """Test the complete document processing pipeline"""
    print("🧪 Testing Document Processing Pipeline")
    print("=" * 50)

    try:
        # Create test document
        test_file_path = await create_test_document()
        print(f"✅ Created test document: {test_file_path}")

        # Test document processing
        processing_request = {
            "chat_id": 1,
            "document_id": 999,  # Test document ID
            "file_path": test_file_path,  # Local file path for testing
            "file_name": "test_document.txt",
            "file_type": "text/plain",
        }

        async with aiohttp.ClientSession() as session:
            # Process document
            print("📄 Processing document...")
            async with session.post(
                "http://localhost:8000/api/documents/process",
                json=processing_request,
                timeout=30,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ Document processing successful!")
                    print(f"   Success: {result.get('success')}")
                    print(f"   Message: {result.get('message')}")

                    stats = result.get("stats", {})
                    if stats:
                        print(f"   Documents: {stats.get('total_documents')}")
                        print(f"   Chunks: {stats.get('total_nodes')}")
                        print(f"   Characters: {stats.get('total_characters')}")
                        print(f"   Avg chunk size: {stats.get('average_node_size')}")
                else:
                    print(f"❌ Document processing failed: {response.status}")
                    print(await response.text())
                    return False

            # Wait a moment for processing to complete
            await asyncio.sleep(2)

            # Test vector store stats
            print("\n📊 Checking vector store stats...")
            async with session.get(
                "http://localhost:8000/api/documents/stats/1"
            ) as response:
                if response.status == 200:
                    stats = await response.json()
                    print("✅ Vector store stats retrieved!")
                    print(f"   Collection: {stats.get('collection_name')}")
                    print(f"   Total chunks: {stats.get('total_chunks')}")
                    print(f"   Total documents: {stats.get('total_documents')}")
                else:
                    print(f"❌ Failed to get stats: {response.status}")

            # Test RAG queries
            print("\n🔍 Testing RAG queries...")
            test_queries = [
                "What is the purpose of this document?",
                "What technology stack is used?",
                "What are the business benefits?",
                "How does the RAG system work?",
            ]

            for i, query in enumerate(test_queries, 1):
                print(f"\n{i}. Query: {query}")

                query_request = {"chat_id": 1, "query": query, "max_results": 5}

                async with session.post(
                    "http://localhost:8000/api/chat/query",
                    json=query_request,
                    timeout=30,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        answer = result.get("answer", "")
                        sources = result.get("sources", [])
                        confidence = result.get("confidence", 0)

                        print(f"   ✅ Answer: {answer[:100]}...")
                        print(f"   📊 Confidence: {confidence:.2f}")
                        print(f"   📚 Sources: {len(sources)}")

                        if sources:
                            for j, source in enumerate(sources[:2], 1):
                                print(
                                    f"      {j}. {source.get('file_name')} (score: {source.get('relevance_score', 0):.3f})"
                                )
                    else:
                        print(f"   ❌ Query failed: {response.status}")

            # Test chat summary
            print("\n📋 Testing chat summary...")
            summary_request = {"chat_id": 1}

            async with session.post(
                "http://localhost:8000/api/chat/summary",
                json=summary_request,
                timeout=30,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    summary = result.get("summary", "")
                    topics = result.get("key_topics", [])
                    doc_count = result.get("document_count", 0)

                    print("✅ Summary generated!")
                    print(f"   📄 Summary: {summary[:150]}...")
                    print(f"   🏷️  Key topics: {', '.join(topics[:3])}")
                    print(f"   📊 Documents: {doc_count}")
                else:
                    print(f"❌ Summary generation failed: {response.status}")

        # Cleanup
        Path(test_file_path).unlink(missing_ok=True)
        print("\n🧹 Cleaned up test files")

        print("\n🎉 Document processing test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        return False


async def main():
    """Run the document processing test"""
    success = await test_document_processing()

    if success:
        print("\n✅ All tests passed! Your RAG system is working correctly.")
        print("\nNext steps:")
        print("1. Test with real PDF/Word documents")
        print("2. Connect to your backend service")
        print("3. Test frontend integration")
        print("4. Deploy to production")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    asyncio.run(main())
