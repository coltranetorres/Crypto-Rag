# Crypto-Rag

A comprehensive Retrieval-Augmented Generation (RAG) system focused on observability and evaluation, built with modern Python frameworks and designed for production use.

## Overview

This project implements a complete RAG pipeline that enables question-answering over document collections using:
- **Strands Framework**: Modern agent framework for building AI applications
- **OpenRouter**: Unified API for accessing multiple LLM and embedding models
- **ChromaDB**: Vector database for efficient similarity search
- **FastAPI**: High-performance backend API
- **Streamlit**: Interactive web frontend
- **RAGAS**: Evaluation framework for RAG systems
- **MLflow**: Experiment tracking and observability

## Features

- 📚 **Document Ingestion**: PDF document processing with semantic chunking
- 🔍 **Vector Search**: Efficient similarity search using ChromaDB
- 🤖 **RAG Agent**: Intelligent query processing with Strands framework
- 🛡️ **Guardrails**: LLM-based content moderation for inputs and outputs
- 📊 **Evaluation**: Comprehensive evaluation pipeline with RAGAS and MLflow
- 🔌 **Dependency Injection**: Clean architecture with DI container
- 📈 **Observability**: Detailed metrics and logging throughout the system
- 🌐 **Web Interface**: User-friendly Streamlit frontend
- 🔧 **REST API**: FastAPI backend for programmatic access

## Architecture

### System Components

```
┌─────────────────────────────────────────┐
│         Streamlit Frontend                │
│      (frontend/app.py)                    │
└──────────────┬────────────────────────────┘
               │ HTTP Requests
┌──────────────▼────────────────────────────┐
│         FastAPI Backend                   │
│       (backend/main.py)                   │
└──────────────┬────────────────────────────┘
               │
┌──────────────▼────────────────────────────┐
│         RAG Agent                         │
│      (src/rag_agent.py)                   │
│  ┌─────────────────────────────────────┐  │
│  │  Strands Agent + Tools             │  │
│  │  - Retrieval Tool                  │  │
│  │  - Guardrails Validator            │  │
│  └─────────────────────────────────────┘  │
└──────────────┬────────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐          ┌──────▼─────┐
│ChromaDB│          │ OpenRouter │
│Vector DB│          │  Provider  │
└────────┘          └────────────┘
```

### Core Modules

#### 1. **RAG Agent** (`src/rag_agent.py`)
The central component that orchestrates retrieval and generation:
- **Agent-based Query**: Uses Strands agent with retrieval tool for intelligent query processing
- **Direct Query**: Faster direct RAG query without agent overhead
- **Guardrails Integration**: Validates inputs and outputs before processing
- **Metadata Tracking**: Comprehensive latency and usage metrics

#### 2. **Document Ingestion** (`src/ingestion.py`)
Handles document processing and vector database population:
- **PDFDocumentLoader**: Extracts text from PDF files with page tracking
- **DocumentProcessor**: Chunks documents using semantic or simple chunking
- **ClusterSemanticChunker**: Advanced semantic chunking (optional)
- **ChromaIngestion**: Batch ingestion with embeddings into ChromaDB

#### 3. **OpenRouter Provider** (`src/openrouter_provider.py`)
Unified interface for LLM and embedding models:
- **OpenRouterProvider**: Manages API calls and token tracking
- **OpenRouterEmbeddings**: Wrapper for embedding generation
- **OpenRouterModel**: Strands-compatible model interface
- **Usage Statistics**: Tracks API calls and token usage

#### 4. **Guardrails** (`src/guardrails.py`)
Content moderation system:
- **GuardrailsValidator**: Validates user inputs and bot responses
- **YAML Configuration**: Flexible guardrail rules via `guardrails.yaml`
- **LLM-based Moderation**: Uses LLM to evaluate content safety

#### 5. **Dependency Injection** (`src/di_container.py`)
Manages component lifecycle and dependencies:
- **DIContainer**: Singleton and factory pattern support
- **Lazy Initialization**: Components created on-demand
- **Settings Integration**: Centralized configuration management

#### 6. **Evaluation** (`src/evaluation.py`)
RAG system evaluation with RAGAS:
- **RAGAS Integration**: Uses RAGAS metrics for evaluation
- **MLflow Tracking**: Experiment tracking and metrics logging
- **Correctness Metric**: Custom discrete metric for answer quality
- **Batch Evaluation**: Processes ground truth datasets

#### 7. **Configuration** (`src/config/`)
Centralized configuration management:
- **Settings**: Pydantic-based settings with environment variable support
- **Logging**: Structured logging configuration

## Installation

### Prerequisites

- Python 3.8+
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Crypto-Rag
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file:
```env
OPENROUTER_API_KEY=your_api_key_here
RAG_LOG_LEVEL=INFO
RAG_CHROMA_DB_PATH=./chroma_db
RAG_COLLECTION_NAME=rag-docs-v1
```

4. **Add documents**
Place PDF files in the `.data` directory:
```bash
mkdir .data
# Copy your PDF files to .data/
```

5. **Run initial setup**
```bash
python setup_day1.py
```

This will:
- Initialize ChromaDB
- Process PDF documents
- Generate embeddings
- Ingest documents into the vector database
- Run test queries

## Usage

### Running the Application

The application consists of two services that need to be running:

#### Step 1: Start the Backend (FastAPI)

In your first terminal window, start the FastAPI backend:

```bash
# Option 1: Using the launcher script
python start_backend.py

# Option 2: Direct uvicorn command
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`

You should see output like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

#### Step 2: Start the Frontend (Streamlit)

In a second terminal window, start the Streamlit frontend:

```bash
# Option 1: Using the launcher script
python start_frontend.py

# Option 2: Direct streamlit command
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

The frontend UI will be available at `http://localhost:8501`

**Important**: Make sure the backend is running before starting the frontend, as the frontend needs to connect to the backend API.

#### Quick Start (Both Services)

If you want to run both services, open two terminal windows:

**Terminal 1 (Backend):**
```bash
uvicorn backend.main:app --reload
```

**Terminal 2 (Frontend):**
```bash
streamlit run frontend/app.py
```

Then open your browser to `http://localhost:8501` to use the application.

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Query
```bash
POST /query
Content-Type: application/json

{
  "question": "What is blockchain?",
  "top_k": 5,
  "use_agent": false
}
```

#### Statistics
```bash
GET /stats
```

### Running Evaluation

```bash
python scripts/run_evaluation.py --qa-csv qa_data.csv
```

Options:
- `--experiment-name`: Custom experiment name
- `--top-k`: Number of documents to retrieve
- `--no-mlflow`: Disable MLflow tracking
- `--mlflow-tracking-uri`: Custom MLflow URI

## Configuration

### Settings (`src/config/settings.py`)

The system uses Pydantic Settings for configuration. All settings can be overridden via environment variables with the `RAG_` prefix:

- `RAG_OPENROUTER_API_KEY`: OpenRouter API key
- `RAG_LLM_MODEL`: LLM model (default: `google/gemini-2.5-flash-lite-preview-09-2025`)
- `RAG_EMBEDDING_MODEL`: Embedding model (default: `thenlper/gte-base`)
- `RAG_CHROMA_DB_PATH`: ChromaDB storage path
- `RAG_COLLECTION_NAME`: Collection name
- `RAG_TOP_K`: Default number of chunks to retrieve
- `RAG_GUARDRAILS_PATH`: Path to guardrails YAML file
- `RAG_LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

### Guardrails Configuration

Edit `guardrails.yaml` to customize content moderation rules:

```yaml
input:
  prompts:
    - task: self_check_input
      content: |
        Your task is to determine whether to block a user request...
        
output:
  prompts:
    - task: self_check_output
      content: |
        Your task is to determine whether the bot response meets...
```

## Project Structure

```
Crypto-Rag/
├── backend/                 # FastAPI backend
│   └── main.py              # API endpoints
├── frontend/                # Streamlit frontend
│   └── app.py              # Web UI
├── src/                     # Core modules
│   ├── config/             # Configuration
│   │   ├── settings.py    # Pydantic settings
│   │   └── logging_config.py
│   ├── rag_agent.py       # RAG agent with Strands
│   ├── ingestion.py       # Document processing
│   ├── openrouter_provider.py  # LLM/embedding provider
│   ├── guardrails.py      # Content moderation
│   ├── evaluation.py      # RAGAS evaluation
│   ├── di_container.py    # Dependency injection
│   └── exceptions.py      # Custom exceptions
├── scripts/                # Utility scripts
│   └── run_evaluation.py  # Evaluation runner
├── tests/                  # Test suite
├── setup_day1.py          # Initial setup script
├── start_backend.py       # Backend launcher
├── start_frontend.py      # Frontend launcher
├── guardrails.yaml        # Guardrails configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Key Design Patterns

### 1. Dependency Injection
The `DIContainer` manages all component dependencies, enabling:
- Easy testing with mock dependencies
- Lazy initialization
- Singleton pattern for shared resources

### 2. Provider Pattern
The `OpenRouterProvider` abstracts LLM/embedding access, allowing:
- Easy model switching
- Usage tracking
- Consistent API interface

### 3. Agent Pattern
The RAG agent uses Strands framework for:
- Tool-based retrieval
- Intelligent query routing
- Extensible architecture

### 4. Factory Pattern
Components are created via factories in the DI container:
- Flexible initialization
- Configuration-driven creation
- Resource management

## Evaluation & Observability

### Metrics Tracked

- **Latency Metrics**:
  - Total query latency
  - Retrieval latency
  - Generation latency

- **Usage Metrics**:
  - Token usage (prompt, completion, total)
  - API call counts
  - Chunks retrieved

- **Quality Metrics**:
  - Correctness scores (pass/fail)
  - Similarity scores
  - Context relevance

### MLflow Integration

Evaluation runs are automatically tracked in MLflow:
- Experiment organization
- Nested runs for each sample
- Parameter and metric logging
- Model response storage

View results:
```bash
mlflow ui --port 5000
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Test modules:
- `test_api.py`: API endpoint tests
- `test_ingestion.py`: Document processing tests
- `test_openrouter_provider.py`: Provider tests
- `test_rag_agent.py`: RAG agent tests

## Development

### Adding New Features

1. **New Document Types**: Extend `PDFDocumentLoader` in `src/ingestion.py`
2. **New Metrics**: Add to `src/evaluation.py` and integrate with RAGAS
3. **New Tools**: Add tools to the Strands agent in `src/rag_agent.py`
4. **New Endpoints**: Add routes to `backend/main.py`

### Logging

The system uses structured logging:
- Log levels: DEBUG, INFO, WARNING, ERROR
- File logging: Optional log file output
- Console logging: Always enabled

### Error Handling

Custom exceptions in `src/exceptions.py`:
- `ProviderError`: LLM provider issues
- `RetrievalError`: Vector DB issues
- `GuardrailBlockedError`: Content moderation blocks
- `ConfigurationError`: Config issues
- `IngestionError`: Document processing issues

## Performance Considerations

- **Batch Processing**: Embeddings generated in batches
- **Caching**: DI container caches singleton instances
- **Rate Limiting**: Built-in delays for API calls
- **Semantic Chunking**: Optional advanced chunking for better retrieval

## Troubleshooting

### Common Issues

1. **Collection not found**
   - Run `setup_day1.py` first
   - Check `RAG_COLLECTION_NAME` setting

2. **OpenRouter API errors**
   - Verify `OPENROUTER_API_KEY` is set
   - Check API quota/limits

3. **No PDFs found**
   - Ensure PDFs are in `.data/` directory
   - Check file permissions

4. **Import errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- **Strands**: Modern agent framework
- **OpenRouter**: Unified LLM API
- **ChromaDB**: Vector database
- **RAGAS**: RAG evaluation framework
- **MLflow**: Experiment tracking
