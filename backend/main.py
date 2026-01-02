"""
FastAPI Backend for RAG Application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from src.di_container import get_container
from src.config.logging_config import setup_logging, get_logger
from src.config.settings import get_settings
from src.exceptions import RetrievalError, ProviderError

load_dotenv()

# Setup logging
import logging
settings = get_settings()
log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
setup_logging(level=log_level, log_file=settings.log_file)
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown"""
    global rag_agent, collection, provider
    
    # Startup
    try:
        logger.info("Initializing RAG agent...")
        
        # Use DI container to resolve dependencies
        container = get_container()
        rag_agent = container.get_rag_agent()
        collection = rag_agent.collection
        provider = rag_agent.provider
        
        logger.info("RAG agent initialized successfully")
        
    except RetrievalError as e:
        logger.error(f"Failed to initialize RAG agent - collection error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Failed to initialize RAG agent: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown (if needed)
    logger.info("Shutting down RAG API")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG agent
rag_agent: Optional[Any] = None
collection: Optional[Any] = None
provider: Optional[Any] = None

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_agent: Optional[bool] = False  # Use agent-based query or direct query

class ContextItem(BaseModel):
    text: str
    source: str
    similarity: float
    page_numbers: Optional[List[int]] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: Optional[List[ContextItem]] = None
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    message: str
    collection_count: Optional[int] = None


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    global collection
    
    if rag_agent is None:
        return HealthResponse(
            status="error",
            message="RAG agent not initialized"
        )
    
    count = collection.count() if collection else None
    return HealthResponse(
        status="healthy",
        message="RAG API is running",
        collection_count=count
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await root()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system
    
    Args:
        request: QueryRequest with question and optional parameters
        
    Returns:
        QueryResponse with answer, contexts, and metadata
    """
    global rag_agent
    
    if rag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="RAG agent not initialized"
        )
    
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        # Use agent-based query or direct query
        if request.use_agent:
            result = rag_agent.query(request.question)
        else:
            result = rag_agent.query_direct(request.question, top_k=request.top_k)
        
        # Convert contexts to ContextItem models if present
        contexts = None
        if result.get('contexts'):
            contexts = [
                ContextItem(
                    text=ctx['text'],
                    source=ctx['source'],
                    similarity=ctx['similarity'],
                    page_numbers=ctx.get('page_numbers', [])
                )
                for ctx in result['contexts']
            ]
        
        return QueryResponse(
            question=result['question'],
            answer=result['answer'],
            contexts=contexts,
            metadata=result['metadata']
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )

@app.get("/stats")
async def get_stats():
    """Get usage statistics from the provider"""
    global provider
    
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail="Provider not initialized"
        )
    
    return provider.get_stats()

