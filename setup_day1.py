"""
Day 1 Setup with Strands + OpenRouter
Complete setup script for RAG system with PDF ingestion and semantic chunking
"""
import os
from dotenv import load_dotenv

from src.di_container import get_container
from src.config.settings import get_settings
from src.config.logging_config import setup_logging, get_logger
from src.ingestion import DocumentProcessor, ChromaIngestion, OpenRouterEmbeddingFunction
from src.openrouter_provider import OpenRouterEmbeddings

load_dotenv()

# Setup logging
settings = get_settings()
import logging
log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
setup_logging(level=log_level, log_file=settings.log_file)
logger = get_logger(__name__)

# Configuration from settings
DATA_DIR = ".data"  # PDF files directory

logger.info("="*60)
logger.info("Day 1 Setup - Strands + OpenRouter")
logger.info("="*60)

# 1. Initialize DI Container
logger.info("[Step 1] Initializing DI container...")
container = get_container()
provider = container.get_provider()

# 2. Initialize ChromaDB
logger.info("[Step 2] Initializing ChromaDB...")
chroma_client = container.get_chroma_client()

# Create collection (delete if exists)
collection_name = settings.collection_name
try:
    chroma_client.delete_collection(collection_name)
    logger.info(f"Deleted existing collection: {collection_name}")
except:
    pass

collection = chroma_client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)
logger.info(f"Created collection: {collection_name}")

# 3. Process and ingest documents
logger.info("[Step 3] Processing PDF documents...")
embedding_wrapper = OpenRouterEmbeddings(provider, settings.embedding_model)
embedding_func = OpenRouterEmbeddingFunction(embedding_wrapper)

# Initialize processor with semantic chunking
processor = DocumentProcessor(
    embedding_function=embedding_func,
    use_semantic_chunking=True,
    max_chunk_size=400
)

documents = processor.load_documents(DATA_DIR)

if not documents:
    print(f"[WARNING] No PDFs found in {DATA_DIR} directory")
    print("[INFO] Please add PDF files to the .data directory")
    exit(1)

chunks = processor.process_documents(documents)

print("\n[Step 4] Ingesting into ChromaDB...")
ingestion = ChromaIngestion(collection, embedding_wrapper, batch_size=50)
total_chunks = ingestion.ingest(chunks)

print(f"\n[SUMMARY] Ingestion Complete:")
print(f"  Documents: {len(documents)}")
print(f"  Chunks: {len(chunks)}")
print(f"  In DB: {total_chunks}")

# Show chunk distribution
chunk_counts = {}
for chunk in chunks:
    source = chunk['metadata']['source']
    chunk_counts[source] = chunk_counts.get(source, 0) + 1

print(f"\n[CHUNK DISTRIBUTION]")
for source, count in chunk_counts.items():
    print(f"  {source}: {count} chunks")

# 4. Initialize RAG Agent
logger.info("[Step 5] Initializing RAG Agent...")
from src.rag_agent import RAGAgent
rag_agent = RAGAgent(
    collection=collection,
    provider=provider,
    top_k=settings.top_k
)

# 5. Test queries
logger.info("[Step 6] Running test queries...")

test_questions = [
    "What are the main topics covered in these documents?",
    "Can you summarize the key points about blockchain technology?",
    "What are the differences between Bitcoin, Ethereum, and Solana?"
]

for q in test_questions:
    logger.info("="*60)
    logger.info(f"[QUESTION] {q}")
    
    # Use direct query for speed
    result = rag_agent.query_direct(q)
    
    logger.info(f"[ANSWER] {result['answer'][:300]}...")
    logger.info(f"[METRICS]")
    logger.info(f"  Total latency: {result['metadata']['total_latency_ms']:.2f}ms")
    logger.info(f"  Retrieval latency: {result['metadata']['retrieval_latency_ms']:.2f}ms")
    logger.info(f"  Generation latency: {result['metadata']['generation_latency_ms']:.2f}ms")
    logger.info(f"  Tokens used: {result['metadata']['usage']['total_tokens']}")
    logger.info(f"  Chunks retrieved: {result['metadata']['chunks_retrieved']}")
    
    if result.get('contexts'):
        logger.info(f"[SOURCES]")
        for i, ctx in enumerate(result['contexts'][:3], 1):
            logger.info(f"  {i}. {ctx['source']} (similarity: {ctx['similarity']:.3f})")

# Print stats
stats = provider.get_stats()
logger.info("="*60)
logger.info("[STATS] OpenRouter Usage:")
for key, value in stats.items():
    logger.info(f"  {key}: {value}")

logger.info("="*60)
logger.info("[SUCCESS] Day 1 Setup Complete!")
logger.info("="*60)
logger.info(f"ChromaDB location: {settings.chroma_db_path}")
logger.info(f"Collection: {collection_name} ({collection.count()} chunks)")
logger.info("Using Strands framework with OpenRouter")
logger.info("Semantic chunking: Enabled")

