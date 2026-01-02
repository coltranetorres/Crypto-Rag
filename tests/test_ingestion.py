"""
Test Data Ingestion Pipeline with PDFs and ClusterSemanticChunker
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.openrouter_provider import OpenRouterProvider, OpenRouterEmbeddings
from src.ingestion import (
    PDFDocumentLoader,
    OpenRouterEmbeddingFunction,
    DocumentProcessor,
    ChromaIngestion
)
import chromadb
from chromadb.config import Settings

load_dotenv()

def test_pdf_loader():
    """Test PDF document loading"""
    print("\n" + "="*60)
    print("[TEST] Testing PDF Document Loader")
    print("="*60)
    
    loader = PDFDocumentLoader(directory="./data")
    documents = loader.load_documents()
    
    if documents:
        print(f"\n[OK] Loaded {len(documents)} document(s)")
        for doc in documents[:2]:  # Show first 2
            print(f"  - {doc['source']}: {doc.get('total_pages', 0)} pages, {len(doc['content'])} chars")
        return True
    else:
        print("\n[INFO] No PDFs found in ./data directory")
        print("       This is OK if you haven't added PDFs yet")
        return True  # Not a failure, just no data


def test_embedding_function():
    """Test OpenRouterEmbeddingFunction"""
    print("\n" + "="*60)
    print("[TEST] Testing OpenRouterEmbeddingFunction")
    print("="*60)
    
    try:
        provider = OpenRouterProvider()
        embedding_wrapper = OpenRouterEmbeddings(provider)
        embedding_func = OpenRouterEmbeddingFunction(embedding_wrapper)
        
        # Test with sample documents
        test_docs = ["This is a test document.", "Another test document."]
        embeddings = embedding_func(test_docs)
        
        assert len(embeddings) == 2, "Should return 2 embeddings"
        assert len(embeddings[0]) == 768, "Each embedding should have 768 dimensions"
        
        print(f"\n[OK] Generated {len(embeddings)} embeddings")
        print(f"[OK] Embedding dimension: {len(embeddings[0])}")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_processor():
    """Test DocumentProcessor with semantic chunking"""
    print("\n" + "="*60)
    print("[TEST] Testing DocumentProcessor")
    print("="*60)
    
    try:
        provider = OpenRouterProvider()
        embedding_wrapper = OpenRouterEmbeddings(provider)
        embedding_func = OpenRouterEmbeddingFunction(embedding_wrapper)
        
        # Test with semantic chunking if available
        processor = DocumentProcessor(
            embedding_function=embedding_func,
            use_semantic_chunking=True,
            max_chunk_size=400
        )
        
        # Create test document
        test_doc = {
            "content": "This is a test document. " * 50,  # Long enough to chunk
            "source": "test.pdf",
            "filepath": "./data/test.pdf",
            "total_pages": 1,
            "pages_with_text": 1
        }
        
        chunks = processor.process_documents([test_doc])
        
        assert len(chunks) > 0, "Should create at least one chunk"
        print(f"\n[OK] Created {len(chunks)} chunk(s)")
        print(f"[OK] Chunking method: {chunks[0]['metadata'].get('chunking_method', 'unknown')}")
        print(f"[OK] First chunk length: {len(chunks[0]['text'])} chars")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test complete ingestion pipeline"""
    print("\n" + "="*60)
    print("[TEST] Testing Full Ingestion Pipeline")
    print("="*60)
    
    try:
        # Initialize components
        provider = OpenRouterProvider()
        embedding_wrapper = OpenRouterEmbeddings(provider)
        embedding_func = OpenRouterEmbeddingFunction(embedding_wrapper)
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db_test",
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        # Create test collection
        try:
            chroma_client.delete_collection("test_ingestion")
        except:
            pass
        
        collection = chroma_client.create_collection(
            name="test_ingestion",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Process documents
        processor = DocumentProcessor(
            embedding_function=embedding_func,
            use_semantic_chunking=True,
            max_chunk_size=400
        )
        
        # Load PDFs (if available)
        documents = processor.load_documents("./data")
        
        if not documents:
            print("\n[INFO] No PDFs found, creating test document")
            # Create a test document
            test_content = "Machine learning is a subset of artificial intelligence. " * 20
            documents = [{
                "content": test_content,
                "source": "test.pdf",
                "filepath": "./data/test.pdf",
                "total_pages": 1,
                "pages_with_text": 1
            }]
        
        # Process into chunks
        chunks = processor.process_documents(documents)
        
        if not chunks:
            print("\n[WARNING] No chunks created")
            return False
        
        # Ingest into ChromaDB
        ingestion = ChromaIngestion(collection, embedding_wrapper, batch_size=10)
        total_count = ingestion.ingest(chunks)
        
        # Verify
        assert total_count == len(chunks), f"Expected {len(chunks)} chunks, got {total_count}"
        
        print(f"\n[OK] Full pipeline successful!")
        print(f"[OK] Ingested {total_count} chunks into ChromaDB")
        
        # Cleanup
        try:
            chroma_client.delete_collection("test_ingestion")
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Data Ingestion Pipeline Test Suite")
    print("="*60)
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\n[WARNING] OPENROUTER_API_KEY not found!")
        return 1
    
    results = []
    
    # Run tests
    results.append(("PDF Loader", test_pdf_loader()))
    results.append(("Embedding Function", test_embedding_function()))
    results.append(("Document Processor", test_document_processor()))
    results.append(("Full Pipeline", test_full_pipeline()))
    
    # Summary
    print("\n" + "="*60)
    print("[SUMMARY] Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"   {status}: {test_name}")
    
    print(f"\n[RESULTS] {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())

