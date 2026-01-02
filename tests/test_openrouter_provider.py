"""
Test OpenRouter Provider - LLM and Embeddings
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.openrouter_provider import OpenRouterProvider, OpenRouterEmbeddings

# Load environment variables
load_dotenv()

def test_llm_completion():
    """Test LLM completion from OpenRouter"""
    print("\n" + "="*60)
    print("[TEST] Testing LLM Completion (OpenRouter)")
    print("="*60)
    
    # Initialize provider
    provider = OpenRouterProvider(
        default_llm_model="google/gemini-2.5-flash-lite-preview-09-2025",
        default_embedding_model="thenlper/gte-base"
    )
    
    # Test message
    messages = [
        {
            "role": "user",
            "content": "Say 'Hello, OpenRouter!' in a creative way. Keep it short."
        }
    ]
    
    try:
        print("\n[SENDING] Sending request to OpenRouter...")
        response = provider.complete(
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        print("\n[SUCCESS] LLM Completion Successful!")
        print(f"[RESPONSE] {response.content}")
        print(f"[MODEL] {response.model}")
        print(f"[USAGE]")
        print(f"   - Prompt tokens: {response.usage['prompt_tokens']}")
        print(f"   - Completion tokens: {response.usage['completion_tokens']}")
        print(f"   - Total tokens: {response.usage['total_tokens']}")
        
        # Verify response structure
        assert response.content is not None, "Response content should not be None"
        assert len(response.content) > 0, "Response content should not be empty"
        assert response.model is not None, "Model name should not be None"
        assert response.usage['total_tokens'] > 0, "Total tokens should be greater than 0"
        
        print("\n[PASS] All LLM completion assertions passed!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] LLM Completion Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embeddings():
    """Test embeddings from OpenRouter"""
    print("\n" + "="*60)
    print("[TEST] Testing Embeddings (OpenRouter)")
    print("="*60)
    
    # Initialize provider
    provider = OpenRouterProvider(
        default_llm_model="google/gemini-2.5-flash-lite-preview-09-2025",
        default_embedding_model="thenlper/gte-base"
    )
    
    # Test texts
    test_texts = [
        "This is a test sentence for embeddings.",
        "Another sentence to test multiple embeddings.",
        "Machine learning and AI are fascinating topics."
    ]
    
    try:
        print(f"\n[SENDING] Generating embeddings for {len(test_texts)} texts...")
        response = provider.embed(texts=test_texts)
        
        print("\n[SUCCESS] Embeddings Generated Successfully!")
        print(f"[MODEL] {response.model}")
        print(f"[COUNT] Number of embeddings: {len(response.embeddings)}")
        print(f"[DIM] Embedding dimension: {len(response.embeddings[0])}")
        print(f"[USAGE]")
        print(f"   - Total tokens: {response.usage.get('total_tokens', 'N/A')}")
        
        # Verify response structure
        assert len(response.embeddings) == len(test_texts), f"Expected {len(test_texts)} embeddings, got {len(response.embeddings)}"
        assert len(response.embeddings[0]) == 768, f"Expected 768 dimensions for gte-base, got {len(response.embeddings[0])}"
        assert all(isinstance(emb, list) for emb in response.embeddings), "All embeddings should be lists"
        assert all(len(emb) == 768 for emb in response.embeddings), "All embeddings should have 768 dimensions"
        
        # Show first embedding stats
        first_embedding = response.embeddings[0]
        print(f"\n[STATS] First embedding statistics:")
        print(f"   - Length: {len(first_embedding)}")
        print(f"   - Min value: {min(first_embedding):.4f}")
        print(f"   - Max value: {max(first_embedding):.4f}")
        print(f"   - Mean value: {sum(first_embedding)/len(first_embedding):.4f}")
        
        print("\n[PASS] All embedding assertions passed!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Embedding Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_wrapper():
    """Test the OpenRouterEmbeddings wrapper"""
    print("\n" + "="*60)
    print("[TEST] Testing OpenRouterEmbeddings Wrapper")
    print("="*60)
    
    try:
        # Initialize provider and wrapper
        provider = OpenRouterProvider(
            default_embedding_model="thenlper/gte-base"
        )
        embedding_wrapper = OpenRouterEmbeddings(provider=provider)
        
        # Test embed_documents
        print("\n[TESTING] embed_documents()...")
        documents = ["Document 1", "Document 2"]
        doc_embeddings = embedding_wrapper.embed_documents(documents)
        
        assert len(doc_embeddings) == 2, "Should return 2 embeddings"
        assert len(doc_embeddings[0]) == 768, "Each embedding should have 768 dimensions"
        print("[PASS] embed_documents() works correctly!")
        
        # Test embed_query
        print("\n[TESTING] embed_query()...")
        query = "What is this about?"
        query_embedding = embedding_wrapper.embed_query(query)
        
        assert isinstance(query_embedding, list), "Query embedding should be a list"
        assert len(query_embedding) == 768, "Query embedding should have 768 dimensions"
        print("[PASS] embed_query() works correctly!")
        
        # Test dimension property
        assert embedding_wrapper.dimension == 768, "Dimension should be 768 for gte-base"
        print(f"[PASS] Dimension property: {embedding_wrapper.dimension}")
        
        print("\n[PASS] All wrapper tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Wrapper Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_stats():
    """Test provider statistics tracking"""
    print("\n" + "="*60)
    print("[TEST] Testing Provider Statistics")
    print("="*60)
    
    try:
        provider = OpenRouterProvider()
        
        # Make some calls
        messages = [{"role": "user", "content": "Test"}]
        provider.complete(messages, max_tokens=10)
        provider.embed(["test text"])
        
        stats = provider.get_stats()
        
        print(f"\n[STATS] Provider Statistics:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
        assert stats['llm_calls'] == 1, "Should have 1 LLM call"
        assert stats['embedding_calls'] == 1, "Should have 1 embedding call"
        assert stats['total_tokens'] > 0, "Should have used some tokens"
        
        print("\n[PASS] Statistics tracking works correctly!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Statistics Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("OpenRouter Provider Test Suite")
    print("="*60)
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\n[WARNING] OPENROUTER_API_KEY not found in environment variables!")
        print("   Please set it in your .env file or environment.")
        print("   Tests will still run but may fail if API key is required.\n")
    else:
        print(f"\n[OK] API Key found: {api_key[:10]}...{api_key[-4:]}\n")
    
    results = []
    
    # Run tests
    results.append(("LLM Completion", test_llm_completion()))
    results.append(("Embeddings", test_embeddings()))
    results.append(("Embedding Wrapper", test_embedding_wrapper()))
    results.append(("Provider Stats", test_provider_stats()))
    
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

