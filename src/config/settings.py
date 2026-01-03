"""
Centralized configuration management using Pydantic Settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # OpenRouter Configuration
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    llm_model: str = "google/gemini-2.5-flash-lite-preview-09-2025"
    embedding_model: str = "thenlper/gte-base"
    site_url: str = "http://localhost:3000"
    site_name: str = "RAG Observability Project"
    
    # ChromaDB Configuration
    chroma_db_path: str = "./chroma_db"
    collection_name: str = "rag-docs-v1"
    
    # RAG Configuration
    top_k: int = 5
    guardrails_path: Optional[str] = "guardrails.yaml"
    
    # MLflow Configuration
    mlflow_tracking_uri: Optional[str] = None  # Defaults to file:./mlruns
    mlflow_experiment_prefix: str = "rag_evaluation"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        case_sensitive=False
    )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance (singleton pattern)
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment (useful for testing)
    
    Returns:
        New Settings instance
    """
    global _settings
    _settings = Settings()
    return _settings

