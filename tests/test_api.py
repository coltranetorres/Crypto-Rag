"""
API endpoint tests for FastAPI backend
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.main import app
from src.rag_agent import RAGAgent


@pytest.fixture
def mock_rag_agent():
    """Create a mock RAG agent"""
    agent = Mock(spec=RAGAgent)
    agent.query_direct.return_value = {
        "question": "test question",
        "answer": "test answer",
        "contexts": [
            {
                "text": "test context",
                "source": "test.pdf",
                "similarity": 0.95,
                "page_numbers": [1]
            }
        ],
        "blocked": False,
        "block_reason": None,
        "metadata": {
            "total_latency_ms": 100.0,
            "retrieval_latency_ms": 50.0,
            "generation_latency_ms": 50.0,
            "model": "test-model",
            "embedding_model": "test-embedding",
            "chunks_retrieved": 1,
            "usage": {"total_tokens": 100}
        }
    }
    agent.query.return_value = agent.query_direct.return_value
    agent.collection = Mock()
    agent.collection.count.return_value = 100
    agent.provider = Mock()
    agent.provider.get_stats.return_value = {
        "llm_calls": 10,
        "embedding_calls": 5,
        "total_tokens": 1000
    }
    return agent


@pytest.fixture
def client(mock_rag_agent):
    """Create test client with mocked dependencies"""
    with patch('backend.main.rag_agent', mock_rag_agent), \
         patch('backend.main.collection', mock_rag_agent.collection), \
         patch('backend.main.provider', mock_rag_agent.provider):
        yield TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["message"] == "RAG API is running"
    assert data["collection_count"] == 100


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_query_endpoint_success(client, mock_rag_agent):
    """Test successful query endpoint"""
    response = client.post(
        "/query",
        json={
            "question": "What is the test question?",
            "top_k": 5,
            "use_agent": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "test question"
    assert data["answer"] == "test answer"
    assert len(data["contexts"]) == 1
    assert data["contexts"][0]["source"] == "test.pdf"
    assert data["metadata"]["total_latency_ms"] == 100.0
    mock_rag_agent.query_direct.assert_called_once()


def test_query_endpoint_empty_question(client):
    """Test query endpoint with empty question"""
    response = client.post(
        "/query",
        json={
            "question": "",
            "top_k": 5
        }
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_query_endpoint_whitespace_question(client):
    """Test query endpoint with whitespace-only question"""
    response = client.post(
        "/query",
        json={
            "question": "   ",
            "top_k": 5
        }
    )
    assert response.status_code == 400


def test_query_endpoint_use_agent(client, mock_rag_agent):
    """Test query endpoint with agent-based query"""
    response = client.post(
        "/query",
        json={
            "question": "test question",
            "use_agent": True
        }
    )
    assert response.status_code == 200
    mock_rag_agent.query.assert_called_once()
    mock_rag_agent.query_direct.assert_not_called()


def test_query_endpoint_custom_top_k(client, mock_rag_agent):
    """Test query endpoint with custom top_k"""
    response = client.post(
        "/query",
        json={
            "question": "test question",
            "top_k": 10
        }
    )
    assert response.status_code == 200
    # Verify top_k was passed
    call_args = mock_rag_agent.query_direct.call_args
    assert call_args[1]["top_k"] == 10


def test_stats_endpoint(client, mock_rag_agent):
    """Test stats endpoint"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "llm_calls" in data
    assert "embedding_calls" in data
    assert "total_tokens" in data
    assert data["llm_calls"] == 10


def test_query_endpoint_no_agent_initialized():
    """Test query endpoint when agent is not initialized"""
    with patch('backend.main.rag_agent', None):
        client = TestClient(app)
        response = client.post(
            "/query",
            json={
                "question": "test question"
            }
        )
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()


def test_stats_endpoint_no_provider():
    """Test stats endpoint when provider is not initialized"""
    with patch('backend.main.provider', None):
        client = TestClient(app)
        response = client.get("/stats")
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()

