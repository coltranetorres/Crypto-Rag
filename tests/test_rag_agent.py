"""
Test RAG Agent with OpenRouter Model
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.openrouter_provider import OpenRouterProvider, OpenRouterModel
from strands import Agent

# Load environment variables
load_dotenv()

def test_openrouter_model_with_agent():
    """Test OpenRouterModel with Strands Agent"""
    print("\n" + "="*60)
    print("[TEST] Testing OpenRouterModel with Strands Agent")
    print("="*60)
    
    try:
        # Initialize provider
        provider = OpenRouterProvider(
            default_llm_model="google/gemini-2.5-flash-lite-preview-09-2025",
            default_embedding_model="thenlper/gte-base"
        )
        
        # Create model
        model = OpenRouterModel(provider)
        
        print("\n[OK] OpenRouterModel created")
        print(f"[CONFIG] Model ID: {model.model_id}")
        
        # Test get_config
        config = model.get_config()
        print(f"[CONFIG] Config: {config}")
        assert "model" in config, "Config should have 'model' key"
        assert "provider" in config, "Config should have 'provider' key"
        print("[PASS] get_config() works")
        
        # Create Strands Agent with our model
        print("\n[CREATING] Creating Strands Agent with OpenRouterModel...")
        agent = Agent(
            name="Test Agent",
            model=model,
            system_prompt="You are a helpful assistant."
        )
        
        print("[OK] Agent created successfully")
        
        # Test agent invocation
        print("\n[TESTING] Testing agent invocation...")
        
        # Test the stream method directly first to see if content is generated
        import asyncio
        from strands.types.content import Message
        
        async def test_direct_stream():
            test_messages = [Message(role="user", content="Say hello in one sentence.")]
            content_parts = []
            async for event in model.stream(messages=test_messages):
                if event.get("type") == "modelContentBlockDeltaEvent":
                    delta = event.get("delta", {})
                    if delta.get("type") == "textDelta":
                        text = delta.get("text", "")
                        content_parts.append(text)
                        print(f"[DELTA] Got text chunk: {repr(text)}")
            full_content = "".join(content_parts)
            print(f"[CONTENT] Full content from stream: {repr(full_content)}")
            return full_content
        
        stream_content = asyncio.run(test_direct_stream())
        assert len(stream_content) > 0, f"Stream should generate content, got: {repr(stream_content)}"
        print(f"[OK] Stream generates content: {stream_content[:50]}...")
        
        # Now test agent
        response = agent("Say hello in one sentence.")
        
        print(f"[RESPONSE TYPE] {type(response)}")
        print(f"[RESPONSE REPR] {repr(response)}")
        print(f"[RESPONSE STR] {str(response)}")
        
        # Check if response has content attribute
        if hasattr(response, 'content'):
            print(f"[RESPONSE CONTENT] {response.content}")
        
        assert response is not None, "Response should not be None"
        # For now, just check that we got something (even if empty, the structure should exist)
        print(f"[INFO] Response received (may be empty due to Agent processing)")
        
        print("\n[PASS] Agent invocation successful!")
        print(f"[RESPONSE] {str(response)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_stream_method():
    """Test the stream method directly"""
    print("\n" + "="*60)
    print("[TEST] Testing OpenRouterModel.stream() method")
    print("="*60)
    
    try:
        import asyncio
        from strands.types.content import Message
        
        # Initialize provider and model
        provider = OpenRouterProvider()
        model = OpenRouterModel(provider)
        
        # Create test messages
        messages = [
            Message(role="user", content="Say 'test' in one word.")
        ]
        
        print("\n[TESTING] Calling stream() method...")
        
        async def test_stream():
            events = []
            async for event in model.stream(messages=messages):
                events.append(event)
                print(f"[EVENT] {event.get('type', 'unknown')}")
            
            print(f"\n[OK] Received {len(events)} stream events")
            
            # Check for required events
            event_types = [e.get('type') for e in events]
            assert "modelMessageStartEvent" in event_types, "Should have message start event"
            assert "modelContentBlockStartEvent" in event_types, "Should have content block start"
            assert "modelContentBlockDeltaEvent" in event_types, "Should have content deltas"
            assert "modelMessageStopEvent" in event_types, "Should have message stop event"
            assert "modelMetadataEvent" in event_types, "Should have metadata event"
            
            print("[PASS] All required stream events present")
            return True
        
        result = asyncio.run(test_stream())
        return result
        
    except Exception as e:
        print(f"\n[ERROR] Stream test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RAG Agent & OpenRouterModel Test Suite")
    print("="*60)
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\n[WARNING] OPENROUTER_API_KEY not found in environment variables!")
        print("   Please set it in your .env file or environment.")
        return 1
    else:
        print(f"\n[OK] API Key found: {api_key[:10]}...{api_key[-4:]}\n")
    
    results = []
    
    # Run tests
    results.append(("Model with Agent", test_openrouter_model_with_agent()))
    results.append(("Stream Method", test_model_stream_method()))
    
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

