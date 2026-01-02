"""
Custom exception hierarchy for the RAG application
"""
from typing import Optional


class RAGException(Exception):
    """Base exception for RAG application"""
    pass


class ProviderError(RAGException):
    """LLM provider errors"""
    def __init__(self, message: str, provider: Optional[str] = None):
        self.provider = provider
        super().__init__(message)


class RetrievalError(RAGException):
    """Vector database retrieval errors"""
    def __init__(self, message: str, collection: Optional[str] = None):
        self.collection = collection
        super().__init__(message)


class GuardrailBlockedError(RAGException):
    """Content blocked by guardrails"""
    def __init__(self, reason: str, content_type: str = "unknown"):
        self.reason = reason
        self.content_type = content_type
        super().__init__(f"Content blocked ({content_type}): {reason}")


class ConfigurationError(RAGException):
    """Configuration-related errors"""
    pass


class IngestionError(RAGException):
    """Document ingestion errors"""
    def __init__(self, message: str, source: Optional[str] = None):
        self.source = source
        super().__init__(message)


class EvaluationError(RAGException):
    """Evaluation pipeline errors"""
    pass

