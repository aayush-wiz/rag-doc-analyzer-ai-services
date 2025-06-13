#!/usr/bin/env python3
"""
Complete Integration Test Script
Tests your full backend ↔ AI services integration
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

# Configuration
BACKEND_URL = "http://localhost:3001"
AI_SERVICE_URL = "http://localhost:8000"

# Test credentials (update these)
TEST_USER = {
    "email": "test@example.com",
    "password": "testpassword123",
    "firstName": "Test",
    "lastName": "User",
}


class IntegrationTester:
    """Complete integration test suite"""

    def __init__(self):
        self.session = None
        self.auth_token = None
        self.test_chat_id = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_ai_service_health(self):
        """Test AI service health check"""
        print("🔍 Testing AI Service Health...")

        try:
            async with self.session.get(f"{AI_SERVICE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ AI Service is healthy")
                    print(f"   Components: {data.get('components', {})}")
                    return True
                else:
                    print(f"❌ AI Service health check failed: {response.status}")
                    return False

        except Exception as e:
            print(f"❌ AI Service connection failed: {str(e)}")
            return False

    async def test_backend_health(self):
        """Test backend health"""
        print("🔍 Testing Backend Health...")

        try:
            async with self.session.get(f"{BACKEND_URL}/api/health") as response:
                if response.status == 200:
                    print("✅ Backend is healthy")
                    return True
                else:
                    print(f"❌ Backend health check failed: {response.status}")
                    return False

        except Exception as e:
            print(f"❌ Backend connection failed: {str(e)}")
            return False

    async def test_user_auth(self):
        """Test user authentication"""
        print("🔍 Testing User Authentication...")

        try:
            # Try to login first
            login_data = {
                "email": TEST_USER["email"],
                "password": TEST_USER["password"],
            }

            async with self.session.post(
                f"{BACKEND_URL}/api/auth/login", json=login_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.auth_token = data.get("token")
                    print("✅ User login successful")
                    return True
                elif response.status == 401:
                    # User doesn't exist, try to signup
                    print("   User doesn't exist, creating account...")
                    return await self._signup_user()
                else:
                    text = await response.text()
                    print(f"❌ Login failed: {response.status} - {text}")
                    return False

        except Exception as e:
            print(f"❌ Authentication error: {str(e)}")
            return False

    async def _signup_user(self):
        """Sign up test user"""
        try:
            async with self.session.post(
                f"{BACKEND_URL}/api/auth/signup", json=TEST_USER
            ) as response:
                if response.status == 201:
                    data = await response.json()
                    self.auth_token = data.get("token")
                    print("✅ User signup successful")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Signup failed: {response.status} - {text}")
                    return False

        except Exception as e:
            print(f"❌ Signup error: {str(e)}")
            return False

    async def test_chat_creation(self):
        """Test chat creation"""
        print("🔍 Testing Chat Creation...")

        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            chat_data = {"title": "Integration Test Chat"}

            async with self.session.post(
                f"{BACKEND_URL}/api/chat", json=chat_data, headers=headers
            ) as response:
                if response.status == 201:
                    data = await response.json()
                    self.test_chat_id = data.get("id")
                    print(f"✅ Chat created successfully (ID: {self.test_chat_id})")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Chat creation failed: {response.status} - {text}")
                    return False

        except Exception as e:
            print(f"❌ Chat creation error: {str(e)}")
            return False

    async def test_ai_service_config(self):
        """Test AI service configuration"""
        print("🔍 Testing AI Service Configuration...")

        try:
            async with self.session.get(f"{AI_SERVICE_URL}/api/config") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ AI Service configuration retrieved")
                    print(f"   Primary LLM: {data.get('llm', {}).get('primary')}")
                    print(
                        f"   Embedding Model: {data.get('document_processing', {}).get('chunk_size')}"
                    )
                    return True
                else:
                    print(f"❌ AI Service config failed: {response.status}")
                    return False

        except Exception as e:
            print(f"❌ AI Service config error: {str(e)}")
            return False

    async def test_document_processing_simulation(self):
        """Test document processing (simulation)"""
        print("🔍 Testing Document Processing...")

        try:
            # Simulate document processing request
            doc_request = {
                "chat_id": self.test_chat_id,
                "document_id": 1,
                "file_path": "https://example.com/test.pdf",
                "file_name": "test.pdf",
                "file_type": "application/pdf",
            }

            async with self.session.post(
                f"{AI_SERVICE_URL}/api/documents/process", json=doc_request
            ) as response:
                if response.status in [200, 422]:  # 422 expected for invalid URL
                    data = await response.json()
                    print("✅ Document processing endpoint accessible")
                    print(
                        f"   Response structure valid: {bool(data.get('success') is not None)}"
                    )
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Document processing failed: {response.status} - {text}")
                    return False

        except Exception as e:
            print(f"❌ Document processing error: {str(e)}")
            return False

    async def test_rag_query_simulation(self):
        """Test RAG query (simulation)"""
        print("🔍 Testing RAG Query...")

        try:
            # Simulate RAG query
            query_request = {
                "chat_id": self.test_chat_id,
                "query": "What is the main topic of the documents?",
                "max_results": 5,
            }

            async with self.session.post(
                f"{AI_SERVICE_URL}/api/chat/query", json=query_request
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ RAG query endpoint accessible")
                    print(f"   Answer provided: {bool(data.get('answer'))}")
                    print(
                        f"   Sources structure: {bool(isinstance(data.get('sources'), list))}"
                    )
                    print(f"   Confidence score: {data.get('confidence', 'N/A')}")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ RAG query failed: {response.status} - {text}")
                    return False

        except Exception as e:
            print(f"❌ RAG query error: {str(e)}")
            return False

    async def test_chat_summary(self):
        """Test chat summary"""
        print("🔍 Testing Chat Summary...")

        try:
            summary_request = {"chat_id": self.test_chat_id}

            async with self.session.post(
                f"{AI_SERVICE_URL}/api/chat/summary", json=summary_request
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Chat summary endpoint accessible")
                    print(f"   Summary provided: {bool(data.get('summary'))}")
                    print(f"   Topics: {data.get('key_topics', [])}")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Chat summary failed: {response.status} - {text}")
                    return False

        except Exception as e:
            print(f"❌ Chat summary error: {str(e)}")
            return False

    async def test_vector_store_stats(self):
        """Test vector store statistics"""
        print("🔍 Testing Vector Store Stats...")

        try:
            async with self.session.get(
                f"{AI_SERVICE_URL}/api/documents/stats/{self.test_chat_id}"
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Vector store stats accessible")
                    print(f"   Collection: {data.get('collection_name')}")
                    print(f"   Documents: {data.get('total_documents', 0)}")
                    print(f"   Chunks: {data.get('total_chunks', 0)}")
                    return True
                else:
                    text = await response.text()
                    print(f"❌ Vector store stats failed: {response.status} - {text}")
                    return False

        except Exception as e:
            print(f"❌ Vector store stats error: {str(e)}")
            return False

    async def run_all_tests(self):
        """Run complete integration test suite"""
        print("🚀 Starting Complete Integration Tests")
        print("=" * 50)

        tests = [
            ("AI Service Health", self.test_ai_service_health),
            ("Backend Health", self.test_backend_health),
            ("User Authentication", self.test_user_auth),
            ("Chat Creation", self.test_chat_creation),
            ("AI Service Config", self.test_ai_service_config),
            ("Document Processing", self.test_document_processing_simulation),
            ("RAG Query", self.test_rag_query_simulation),
            ("Chat Summary", self.test_chat_summary),
            ("Vector Store Stats", self.test_vector_store_stats),
        ]

        results = []
        for test_name, test_func in tests:
            start_time = time.time()

            try:
                result = await test_func()
                duration = time.time() - start_time
                results.append((test_name, result, duration))

            except Exception as e:
                duration = time.time() - start_time
                print(f"❌ {test_name} crashed: {str(e)}")
                results.append((test_name, False, duration))

            print()  # Empty line between tests

        # Summary
        print("📊 Test Results Summary")
        print("=" * 50)

        passed = sum(1 for _, result, _ in results if result)
        total = len(results)

        for test_name, result, duration in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name:<25} ({duration:.2f}s)")

        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            print("\n🎉 All tests passed! Your integration is working perfectly!")
        else:
            print(
                f"\n⚠️  {total-passed} tests failed. Check the logs above for details."
            )

        return passed == total


async def main():
    """Main test function"""
    print("🧪 DocAnalyzer Integration Test Suite")
    print("Testing Backend ↔ AI Services Integration")
    print("=" * 50)

    async with IntegrationTester() as tester:
        success = await tester.run_all_tests()

    if success:
        print("\n✅ Integration test completed successfully!")
        print("Your backend and AI services are properly connected.")
    else:
        print("\n❌ Integration test failed!")
        print("Please check the error messages above and fix the issues.")


if __name__ == "__main__":
    asyncio.run(main())
