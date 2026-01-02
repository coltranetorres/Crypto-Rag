"""
Dependency Injection Container for RAG Application
Manages lifecycle and dependencies of application components
"""
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from src.config.logging_config import get_logger
from src.config.settings import get_settings, Settings
from src.openrouter_provider import OpenRouterProvider, OpenRouterEmbeddings
from src.rag_agent import RAGAgent
from src.guardrails import GuardrailsValidator
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.exceptions import RetrievalError, ConfigurationError

logger = get_logger(__name__)


class DIContainer:
    """
    Dependency Injection Container
    Manages component lifecycle and dependency resolution
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize DI container
        
        Args:
            settings: Optional settings instance (creates new if not provided)
        """
        self.settings = settings or get_settings()
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._initialized = False
    
    def register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton instance"""
        self._instances[name] = instance
        logger.debug(f"Registered singleton: {name}")
    
    def register_factory(self, name: str, factory: Callable) -> None:
        """Register a factory function for lazy initialization"""
        self._factories[name] = factory
        logger.debug(f"Registered factory: {name}")
    
    def get(self, name: str) -> Any:
        """
        Get a dependency by name
        
        Args:
            name: Dependency name
            
        Returns:
            Dependency instance
            
        Raises:
            KeyError: If dependency not found
        """
        # Check if already instantiated
        if name in self._instances:
            return self._instances[name]
        
        # Check if factory exists
        if name in self._factories:
            instance = self._factories[name]()
            self._instances[name] = instance  # Cache as singleton
            return instance
        
        raise KeyError(f"Dependency '{name}' not found. Register it first.")
    
    def has(self, name: str) -> bool:
        """Check if dependency is registered"""
        return name in self._instances or name in self._factories
    
    def clear(self) -> None:
        """Clear all registered dependencies (useful for testing)"""
        self._instances.clear()
        self._factories.clear()
        self._initialized = False
        logger.debug("DI container cleared")
    
    # Convenience methods for common dependencies
    
    def get_settings(self) -> Settings:
        """Get settings instance"""
        return self.settings
    
    def get_provider(self) -> OpenRouterProvider:
        """Get or create OpenRouter provider"""
        if not self.has("provider"):
            self.register_factory("provider", self._create_provider)
        return self.get("provider")
    
    def get_chroma_client(self) -> chromadb.PersistentClient:
        """Get or create ChromaDB client"""
        if not self.has("chroma_client"):
            self.register_factory("chroma_client", self._create_chroma_client)
        return self.get("chroma_client")
    
    def get_collection(self, collection_name: Optional[str] = None) -> Any:
        """Get or create ChromaDB collection"""
        name = collection_name or self.settings.collection_name
        cache_key = f"collection_{name}"
        
        if not self.has(cache_key):
            def factory():
                client = self.get_chroma_client()
                try:
                    collection = client.get_collection(name)
                    count = collection.count()
                    logger.info(f"Loaded collection: {name} ({count} chunks)")
                    return collection
                except Exception as e:
                    raise RetrievalError(
                        f"Collection '{name}' not found. Please run setup_day1.py first.",
                        collection=name
                    ) from e
            
            self.register_factory(cache_key, factory)
        
        return self.get(cache_key)
    
    def get_guardrails_validator(self) -> Optional[GuardrailsValidator]:
        """Get or create guardrails validator"""
        if not self.has("guardrails_validator"):
            def factory():
                provider = self.get_provider()
                guardrails_path = self.settings.guardrails_path
                if guardrails_path:
                    return GuardrailsValidator(provider, guardrails_path)
                return None
            
            self.register_factory("guardrails_validator", factory)
        
        return self.get("guardrails_validator")
    
    def get_rag_agent(
        self,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> RAGAgent:
        """Get or create RAG agent with all dependencies"""
        cache_key = f"rag_agent_{collection_name or self.settings.collection_name}_{top_k or self.settings.top_k}"
        
        if not self.has(cache_key):
            def factory():
                provider = self.get_provider()
                collection = self.get_collection(collection_name)
                guardrails = self.get_guardrails_validator()
                agent_top_k = top_k if top_k is not None else self.settings.top_k
                
                return RAGAgent(
                    collection=collection,
                    provider=provider,
                    top_k=agent_top_k,
                    guardrails=guardrails
                )
            
            self.register_factory(cache_key, factory)
        
        return self.get(cache_key)
    
    # Private factory methods
    
    def _create_provider(self) -> OpenRouterProvider:
        """Create OpenRouter provider"""
        provider = OpenRouterProvider(
            api_key=self.settings.openrouter_api_key,
            site_url=self.settings.site_url,
            site_name=self.settings.site_name,
            default_llm_model=self.settings.llm_model,
            default_embedding_model=self.settings.embedding_model
        )
        logger.info(f"OpenRouter Provider initialized: LLM={provider.default_llm_model}, Embeddings={provider.default_embedding_model}")
        return provider
    
    def _create_chroma_client(self) -> chromadb.PersistentClient:
        """Create ChromaDB client"""
        client = chromadb.PersistentClient(
            path=self.settings.chroma_db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        logger.info(f"ChromaDB client initialized: path={self.settings.chroma_db_path}")
        return client


# Global container instance (can be overridden for testing)
_container: Optional[DIContainer] = None


def get_container(settings: Optional[Settings] = None) -> DIContainer:
    """
    Get the global DI container instance
    
    Args:
        settings: Optional settings to use (creates new container if different)
        
    Returns:
        DIContainer instance
    """
    global _container
    if _container is None or (settings is not None and _container.settings != settings):
        _container = DIContainer(settings)
    return _container


def reset_container() -> None:
    """Reset the global container (useful for testing)"""
    global _container
    _container = None


@contextmanager
def container_scope(settings: Optional[Settings] = None):
    """
    Context manager for container scope (useful for testing)
    
    Usage:
        with container_scope():
            container = get_container()
            # Use container
    """
    global _container
    old_container = _container
    _container = DIContainer(settings)
    try:
        yield _container
    finally:
        _container = old_container

