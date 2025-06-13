#!/usr/bin/env python3
"""
ChromaDB Database Explorer
View and explore all data stored in ChromaDB
Usage: python scripts/explore_chromadb.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    import chromadb
    from chromadb.config import Settings
    import pandas as pd
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install chromadb python-dotenv pandas")
    sys.exit(1)


class ChromaDBExplorer:
    """Explore ChromaDB database contents"""

    def __init__(self):
        # Load environment
        load_dotenv()

        # Setup ChromaDB client
        self.chroma_path = Path("chroma_db")

        if not self.chroma_path.exists():
            print(f"❌ ChromaDB directory not found: {self.chroma_path}")
            print("Make sure your AI service has been running and created data")
            sys.exit(1)

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False,  # Safety: don't allow reset in explorer
                ),
            )
            print(f"✅ Connected to ChromaDB at: {self.chroma_path.absolute()}")
        except Exception as e:
            print(f"❌ Failed to connect to ChromaDB: {e}")
            sys.exit(1)

    def show_database_overview(self) -> None:
        """Show database overview"""
        print("\n" + "=" * 60)
        print("📊 CHROMADB DATABASE OVERVIEW")
        print("=" * 60)

        try:
            # Basic info
            version = self.client.get_version()
            collections = self.client.list_collections()

            print(f"🔧 ChromaDB Version: {version}")
            print(f"📁 Database Path: {self.chroma_path.absolute()}")
            print(f"📚 Total Collections: {len(collections)}")

            # Database size
            size_bytes = sum(
                f.stat().st_size for f in self.chroma_path.rglob("*") if f.is_file()
            )
            size_mb = size_bytes / (1024 * 1024)
            print(f"💾 Database Size: {size_mb:.2f} MB")

            # Collection summary
            total_documents = 0
            for collection in collections:
                count = collection.count()
                total_documents += count

            print(f"📄 Total Documents: {total_documents}")

        except Exception as e:
            print(f"❌ Error getting overview: {e}")

    def list_all_collections(self) -> List:
        """List all collections with details"""
        print("\n" + "=" * 60)
        print("📚 ALL COLLECTIONS")
        print("=" * 60)

        try:
            collections = self.client.list_collections()

            if not collections:
                print("📭 No collections found")
                return []

            collection_data = []
            for i, collection in enumerate(collections, 1):
                count = collection.count()
                metadata = collection.metadata or {}

                print(f"\n{i}. Collection: {collection.name}")
                print(f"   📄 Documents: {count}")
                print(f"   🏷️  Metadata: {metadata}")
                print(f"   🆔 ID: {collection.id}")

                collection_data.append(
                    {
                        "name": collection.name,
                        "count": count,
                        "metadata": metadata,
                        "id": collection.id,
                    }
                )

            return collection_data

        except Exception as e:
            print(f"❌ Error listing collections: {e}")
            return []

    def explore_collection(self, collection_name: str, limit: int = 10) -> None:
        """Explore a specific collection"""
        print(f"\n" + "=" * 60)
        print(f"🔍 EXPLORING COLLECTION: {collection_name}")
        print("=" * 60)

        try:
            collection = self.client.get_collection(collection_name)

            # Get basic info
            count = collection.count()
            metadata = collection.metadata or {}

            print(f"📄 Total Documents: {count}")
            print(f"🏷️  Collection Metadata: {metadata}")

            if count == 0:
                print("📭 Collection is empty")
                return

            # Get sample documents
            print(f"\n📋 SAMPLE DOCUMENTS (showing up to {limit}):")
            print("-" * 40)

            results = collection.get(
                limit=min(limit, count),
                include=["documents", "metadatas", "embeddings"],
            )

            for i, doc_id in enumerate(results["ids"]):
                print(f"\n{i+1}. Document ID: {doc_id}")

                # Document content
                if results["documents"] and i < len(results["documents"]):
                    doc_content = results["documents"][i]
                    if doc_content:
                        # Truncate long documents
                        display_content = (
                            doc_content[:200] + "..."
                            if len(doc_content) > 200
                            else doc_content
                        )
                        print(f"   📝 Content: {display_content}")

                # Metadata
                if results["metadatas"] and i < len(results["metadatas"]):
                    doc_metadata = results["metadatas"][i]
                    if doc_metadata:
                        print(f"   🏷️  Metadata: {doc_metadata}")

                # Embedding info
                if results["embeddings"] and i < len(results["embeddings"]):
                    embedding = results["embeddings"][i]
                    if embedding:
                        print(f"   🧮 Embedding: {len(embedding)} dimensions")

        except Exception as e:
            print(f"❌ Error exploring collection '{collection_name}': {e}")

    def search_documents(
        self, collection_name: str, query_text: str, n_results: int = 5
    ) -> None:
        """Search documents in a collection"""
        print(f"\n" + "=" * 60)
        print(f"🔎 SEARCHING IN: {collection_name}")
        print(f"🔍 Query: '{query_text}'")
        print("=" * 60)

        try:
            collection = self.client.get_collection(collection_name)

            # Perform text search (if collection supports it)
            # Note: This requires the collection to have been set up with appropriate embedding function
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            if not results["ids"][0]:
                print("🔍 No results found")
                return

            print(f"📊 Found {len(results['ids'][0])} results:\n")

            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else "N/A"

                print(f"{i+1}. Document ID: {doc_id}")
                print(f"   📏 Distance: {distance}")

                # Document content
                if results["documents"][0] and i < len(results["documents"][0]):
                    doc_content = results["documents"][0][i]
                    if doc_content:
                        display_content = (
                            doc_content[:300] + "..."
                            if len(doc_content) > 300
                            else doc_content
                        )
                        print(f"   📝 Content: {display_content}")

                # Metadata
                if results["metadatas"][0] and i < len(results["metadatas"][0]):
                    doc_metadata = results["metadatas"][0][i]
                    if doc_metadata:
                        print(f"   🏷️  Metadata: {doc_metadata}")

                print()

        except Exception as e:
            print(f"❌ Error searching collection '{collection_name}': {e}")

    def export_collection_data(
        self, collection_name: str, output_file: str = None
    ) -> None:
        """Export collection data to JSON"""
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()

            if count == 0:
                print(f"📭 Collection '{collection_name}' is empty, nothing to export")
                return

            print(f"📤 Exporting {count} documents from '{collection_name}'...")

            # Get all data
            results = collection.get(include=["documents", "metadatas", "embeddings"])

            # Prepare export data
            export_data = {
                "collection_name": collection_name,
                "collection_metadata": collection.metadata,
                "export_timestamp": datetime.now().isoformat(),
                "document_count": count,
                "documents": [],
            }

            for i, doc_id in enumerate(results["ids"]):
                doc_data = {
                    "id": doc_id,
                    "content": (
                        results["documents"][i] if results["documents"] else None
                    ),
                    "metadata": (
                        results["metadatas"][i] if results["metadatas"] else None
                    ),
                    "embedding_dimensions": (
                        len(results["embeddings"][i])
                        if results["embeddings"] and results["embeddings"][i]
                        else 0
                    ),
                }
                export_data["documents"].append(doc_data)

            # Save to file
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"chromadb_export_{collection_name}_{timestamp}.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Exported to: {output_file}")
            print(f"📊 Exported {len(export_data['documents'])} documents")

        except Exception as e:
            print(f"❌ Error exporting collection '{collection_name}': {e}")

    def interactive_explorer(self) -> None:
        """Interactive exploration mode"""
        print("\n" + "=" * 60)
        print("🎮 INTERACTIVE EXPLORER MODE")
        print("=" * 60)
        print("Commands:")
        print("  1 - List all collections")
        print("  2 - Explore collection")
        print("  3 - Search in collection")
        print("  4 - Export collection")
        print("  5 - Database overview")
        print("  q - Quit")
        print("-" * 60)

        while True:
            try:
                choice = input("\n🎯 Enter command: ").strip().lower()

                if choice == "q":
                    print("👋 Goodbye!")
                    break
                elif choice == "1":
                    self.list_all_collections()
                elif choice == "2":
                    collections = self.client.list_collections()
                    if not collections:
                        print("📭 No collections found")
                        continue

                    print("\nAvailable collections:")
                    for i, col in enumerate(collections):
                        print(f"  {i+1}. {col.name}")

                    try:
                        col_choice = int(input("Select collection number: ")) - 1
                        if 0 <= col_choice < len(collections):
                            limit = int(
                                input("How many documents to show? (default 10): ")
                                or 10
                            )
                            self.explore_collection(collections[col_choice].name, limit)
                        else:
                            print("❌ Invalid collection number")
                    except ValueError:
                        print("❌ Please enter a valid number")

                elif choice == "3":
                    collections = self.client.list_collections()
                    if not collections:
                        print("📭 No collections found")
                        continue

                    print("\nAvailable collections:")
                    for i, col in enumerate(collections):
                        print(f"  {i+1}. {col.name}")

                    try:
                        col_choice = int(input("Select collection number: ")) - 1
                        if 0 <= col_choice < len(collections):
                            query_text = input("Enter search query: ").strip()
                            if query_text:
                                n_results = int(
                                    input("Number of results (default 5): ") or 5
                                )
                                self.search_documents(
                                    collections[col_choice].name, query_text, n_results
                                )
                            else:
                                print("❌ Please enter a search query")
                        else:
                            print("❌ Invalid collection number")
                    except ValueError:
                        print("❌ Please enter valid numbers")

                elif choice == "4":
                    collections = self.client.list_collections()
                    if not collections:
                        print("📭 No collections found")
                        continue

                    print("\nAvailable collections:")
                    for i, col in enumerate(collections):
                        print(f"  {i+1}. {col.name}")

                    try:
                        col_choice = int(input("Select collection number: ")) - 1
                        if 0 <= col_choice < len(collections):
                            output_file = input(
                                "Output file (press Enter for auto-generated): "
                            ).strip()
                            self.export_collection_data(
                                collections[col_choice].name,
                                output_file if output_file else None,
                            )
                        else:
                            print("❌ Invalid collection number")
                    except ValueError:
                        print("❌ Please enter a valid number")

                elif choice == "5":
                    self.show_database_overview()
                else:
                    print("❌ Invalid command. Use 1-5 or 'q'")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break


def main():
    """Main function"""
    print("=" * 60)
    print("🗃️  ChromaDB Database Explorer")
    print("=" * 60)

    explorer = ChromaDBExplorer()

    # Show overview first
    explorer.show_database_overview()
    collections = explorer.list_all_collections()

    if collections:
        print(f"\n💡 Found {len(collections)} collections with data!")

        # Ask if user wants interactive mode
        response = (
            input("\n🎮 Enter interactive explorer mode? (y/n): ").strip().lower()
        )
        if response in ["y", "yes"]:
            explorer.interactive_explorer()
        else:
            print("\n📋 Quick exploration of first collection:")
            if collections:
                explorer.explore_collection(collections[0]["name"], limit=5)
    else:
        print(
            "\n📭 No collections found. Make sure your AI service has processed some documents."
        )


if __name__ == "__main__":
    main()
