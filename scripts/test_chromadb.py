#!/usr/bin/env python3
"""
ChromaDB Test Script
Tests ChromaDB functionality and connection
Usage: python scripts/test_chromadb.py
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    import chromadb
    from chromadb.config import Settings
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages: pip install chromadb python-dotenv numpy")
    sys.exit(1)


class ChromaDBTester:
    """Test ChromaDB functionality"""

    def __init__(self):
        # Load environment
        load_dotenv()

        # Setup ChromaDB client
        self.chroma_path = Path("chroma_db")
        self.chroma_path.mkdir(exist_ok=True)

        self.client = None
        self.test_collection = None

    def test_basic_connection(self) -> bool:
        """Test basic ChromaDB connection"""
        print("🔍 Testing ChromaDB basic connection...")

        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Test basic operations
            version = self.client.get_version()
            print(f"✅ ChromaDB connected successfully")
            print(f"   Version: {version}")
            print(f"   Database path: {self.chroma_path.absolute()}")

            return True

        except Exception as e:
            print(f"❌ ChromaDB connection failed: {str(e)}")
            traceback.print_exc()
            return False

    def test_collection_operations(self) -> bool:
        """Test collection CRUD operations"""
        print("\n🔍 Testing collection operations...")

        try:
            collection_name = f"test_collection_{int(datetime.now().timestamp())}"

            # Create collection
            self.test_collection = self.client.create_collection(
                name=collection_name, metadata={"description": "Test collection"}
            )
            print(f"✅ Created collection: {collection_name}")

            # List collections
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            print(f"✅ Listed collections: {len(collections)} total")
            print(f"   Collections: {collection_names}")

            # Get collection info
            count = self.test_collection.count()
            print(f"✅ Collection count: {count}")

            return True

        except Exception as e:
            print(f"❌ Collection operations failed: {str(e)}")
            traceback.print_exc()
            return False

    def test_document_operations(self) -> bool:
        """Test document add/query operations"""
        print("\n🔍 Testing document operations...")

        try:
            if not self.test_collection:
                print("❌ No test collection available")
                return False

            # Sample documents
            documents = [
                "This is a test document about artificial intelligence.",
                "Machine learning is a subset of AI that focuses on algorithms.",
                "Natural language processing helps computers understand human language.",
                "Vector databases are useful for similarity search.",
                "ChromaDB is a vector database for AI applications.",
            ]

            # Sample embeddings (random for testing)
            embeddings = [np.random.random(384).tolist() for _ in documents]

            # Document IDs and metadata
            ids = [f"doc_{i}" for i in range(len(documents))]
            metadatas = [{"type": "test", "index": i} for i in range(len(documents))]

            # Add documents
            self.test_collection.add(
                embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids
            )
            print(f"✅ Added {len(documents)} documents")

            # Check count
            count = self.test_collection.count()
            print(f"✅ Collection now contains {count} documents")

            # Test query
            query_embedding = np.random.random(384).tolist()
            results = self.test_collection.query(
                query_embeddings=[query_embedding], n_results=3
            )

            print(f"✅ Query returned {len(results['documents'][0])} results")
            print(f"   Sample result IDs: {results['ids'][0]}")

            # Test get by ID
            get_results = self.test_collection.get(
                ids=["doc_0", "doc_1"], include=["documents", "metadatas"]
            )
            print(f"✅ Retrieved {len(get_results['ids'])} documents by ID")

            return True

        except Exception as e:
            print(f"❌ Document operations failed: {str(e)}")
            traceback.print_exc()
            return False

    def test_advanced_features(self) -> bool:
        """Test advanced ChromaDB features"""
        print("\n🔍 Testing advanced features...")

        try:
            if not self.test_collection:
                print("❌ No test collection available")
                return False

            # Test metadata filtering
            filtered_results = self.test_collection.get(where={"type": "test"}, limit=2)
            print(
                f"✅ Metadata filtering returned {len(filtered_results['ids'])} results"
            )

            # Test update
            self.test_collection.update(
                ids=["doc_0"], metadatas=[{"type": "test", "updated": True}]
            )
            print("✅ Updated document metadata")

            # Test upsert
            self.test_collection.upsert(
                ids=["doc_new"],
                embeddings=[np.random.random(384).tolist()],
                documents=["This is a new upserted document"],
                metadatas=[{"type": "upsert"}],
            )
            print("✅ Upserted new document")

            # Final count
            final_count = self.test_collection.count()
            print(f"✅ Final collection count: {final_count}")

            return True

        except Exception as e:
            print(f"❌ Advanced features test failed: {str(e)}")
            traceback.print_exc()
            return False

    def test_persistence(self) -> bool:
        """Test data persistence"""
        print("\n🔍 Testing data persistence...")

        try:
            # Get current collection name
            collection_name = (
                self.test_collection.name if self.test_collection else None
            )
            if not collection_name:
                print("❌ No collection to test persistence")
                return False

            original_count = self.test_collection.count()

            # Close and reopen client
            self.client = None
            self.test_collection = None

            # Reinitialize
            self.client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Get collection back
            self.test_collection = self.client.get_collection(collection_name)
            new_count = self.test_collection.count()

            if original_count == new_count:
                print(f"✅ Data persisted correctly ({new_count} documents)")
                return True
            else:
                print(f"❌ Data persistence failed: {original_count} → {new_count}")
                return False

        except Exception as e:
            print(f"❌ Persistence test failed: {str(e)}")
            traceback.print_exc()
            return False

    def cleanup(self) -> None:
        """Clean up test data"""
        print("\n🧹 Cleaning up test data...")

        try:
            if self.client and self.test_collection:
                # Delete test collection
                self.client.delete_collection(self.test_collection.name)
                print(f"✅ Deleted test collection: {self.test_collection.name}")

        except Exception as e:
            print(f"⚠️  Cleanup warning: {str(e)}")

    def get_database_info(self) -> None:
        """Get ChromaDB database information"""
        print("\n📊 ChromaDB Database Information:")

        try:
            if not self.client:
                print("❌ No client available")
                return

            # List all collections
            collections = self.client.list_collections()
            print(f"   Total collections: {len(collections)}")

            for collection in collections:
                count = collection.count()
                print(f"   - {collection.name}: {count} documents")

            # Database size
            db_path = self.chroma_path
            if db_path.exists():
                size_bytes = sum(
                    f.stat().st_size for f in db_path.rglob("*") if f.is_file()
                )
                size_mb = size_bytes / (1024 * 1024)
                print(f"   Database size: {size_mb:.2f} MB")

        except Exception as e:
            print(f"❌ Failed to get database info: {str(e)}")


def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 ChromaDB Test Suite")
    print("=" * 60)

    tester = ChromaDBTester()

    try:
        # Run tests
        tests = [
            ("Basic Connection", tester.test_basic_connection),
            ("Collection Operations", tester.test_collection_operations),
            ("Document Operations", tester.test_document_operations),
            ("Advanced Features", tester.test_advanced_features),
            ("Data Persistence", tester.test_persistence),
        ]

        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            results.append((test_name, result))

        # Get database info
        tester.get_database_info()

        # Summary
        print("\n" + "=" * 60)
        print("📋 Test Results Summary:")
        print("=" * 60)

        passed = 0
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
            if result:
                passed += 1

        print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")

        if passed == len(results):
            print("🎉 All tests passed! ChromaDB is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the errors above.")

    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        traceback.print_exc()
    finally:
        # Cleanup
        tester.cleanup()
        print("\n👋 Test complete!")


if __name__ == "__main__":
    main()
