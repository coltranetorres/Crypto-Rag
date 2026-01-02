"""
RAG Evaluation Runner Script
Runs full evaluation on qa_data.csv with RAGAS and MLflow integration
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.di_container import get_container
from src.config.settings import get_settings
from src.config.logging_config import setup_logging, get_logger
from src.evaluation import (
    load_qa_dataset,
    create_ragas_dataset,
    evaluate_rag,
    save_results
)

load_dotenv()

# Setup logging
settings = get_settings()
log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
setup_logging(level=log_level, log_file=settings.log_file)
logger = get_logger(__name__)

# Configuration
DEFAULT_QA_CSV = "qa_data.csv"
DEFAULT_EXPERIMENT_NAME = None  # Will auto-generate with timestamp


async def run_evaluation(
    qa_csv_path: str = DEFAULT_QA_CSV,
    experiment_name: str = None,
    top_k: int = 5,
    use_mlflow: bool = True,
    mlflow_tracking_uri: str = None,
    collection_name: str = COLLECTION_NAME,
    site_url: str = "http://localhost:8000",
    site_name: str = "RAG Evaluation",
    llm_model: str = "google/gemini-2.5-flash-lite-preview-09-2025",
    embedding_model: str = "thenlper/gte-base"
):
    """
    Run full RAG evaluation on ground truth dataset
    
    Args:
        qa_csv_path: Path to Q&A CSV file
        experiment_name: Optional experiment name (auto-generated if None)
        top_k: Number of documents to retrieve
        use_mlflow: Whether to use MLflow for tracking
        mlflow_tracking_uri: Optional MLflow tracking URI
        collection_name: ChromaDB collection name
        site_url: OpenRouter site URL
        site_name: OpenRouter site name
        llm_model: LLM model to use
        embedding_model: Embedding model to use
    """
    logger.info("="*60)
    logger.info("RAG Evaluation Runner")
    logger.info("="*60)
    
    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{settings.mlflow_experiment_prefix}_{timestamp}"
    
    try:
        # 1-3. Initialize RAG Agent using DI container
        logger.info("[Step 1-3] Initializing RAG system...")
        container = get_container()
        rag_agent = container.get_rag_agent(
            collection_name=collection_name,
            top_k=top_k
        )
        provider = rag_agent.provider
        logger.info("RAG Agent initialized")
        
        # 4. Load Ground Truth Dataset
        logger.info(f"[Step 4] Loading ground truth dataset from {qa_csv_path}...")
        if not os.path.exists(qa_csv_path):
            logger.error(f"Dataset not found: {qa_csv_path}")
            return None
        
        df = load_qa_dataset(qa_csv_path)
        logger.info(f"Loaded {len(df)} Q&A pairs")
        
        # 5. Create RAGAS Dataset
        logger.info("[Step 5] Creating RAGAS dataset...")
        dataset = create_ragas_dataset(df)
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # 6. Run Evaluation
        logger.info(f"[Step 6] Running evaluation (experiment: {experiment_name})...")
        logger.info(f"  - Samples: {len(dataset)}")
        logger.info(f"  - Top K: {top_k}")
        logger.info(f"  - MLflow: {'Enabled' if use_mlflow else 'Disabled'}")
        
        results = await evaluate_rag(
            rag_agent=rag_agent,
            dataset=dataset,
            experiment_name=experiment_name,
            top_k=top_k,
            openrouter_provider=provider,
            use_mlflow=use_mlflow,
            mlflow_tracking_uri=mlflow_tracking_uri
        )
        
        # 7. Save Results
        logger.info("[Step 7] Saving results...")
        filepath = save_results(results, experiment_name=experiment_name)
        
        # 8. Print Summary
        logger.info("="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        if results:
            pass_count = sum(1 for r in results if r.get("correctness_score") == "pass")
            fail_count = sum(1 for r in results if r.get("correctness_score") == "fail")
            error_count = sum(1 for r in results if r.get("correctness_score") == "error")
            total_count = len(results)
            pass_rate = (pass_count / total_count) * 100 if total_count > 0 else 0
            
            logger.info(f"Total Samples: {total_count}")
            logger.info(f"Passed: {pass_count} ({pass_rate:.1f}%)")
            logger.info(f"Failed: {fail_count}")
            logger.info(f"Errors: {error_count}")
            logger.info(f"Results CSV: {filepath}")
            if use_mlflow:
                logger.info(f"MLflow Experiment: {experiment_name}")
                logger.info(f"View in MLflow UI: mlflow ui --port 5000")
        logger.info("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return None


def main():
    """Main entry point for the evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation on ground truth dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with default settings
  python scripts/run_evaluation.py
  
  # Run with custom experiment name
  python scripts/run_evaluation.py --experiment-name "baseline_v1"
  
  # Run without MLflow
  python scripts/run_evaluation.py --no-mlflow
  
  # Run with custom dataset
  python scripts/run_evaluation.py --qa-csv "custom_data.csv"
        """
    )
    
    parser.add_argument(
        "--qa-csv",
        type=str,
        default=DEFAULT_QA_CSV,
        help=f"Path to Q&A CSV file (default: {DEFAULT_QA_CSV})"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking"
    )
    
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: local file store)"
    )
    
    parser.add_argument(
        "--collection-name",
        type=str,
        default=COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {COLLECTION_NAME})"
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        default="google/gemini-2.5-flash-lite-preview-09-2025",
        help="LLM model to use"
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="thenlper/gte-base",
        help="Embedding model to use"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = asyncio.run(run_evaluation(
        qa_csv_path=args.qa_csv,
        experiment_name=args.experiment_name,
        top_k=args.top_k,
        use_mlflow=not args.no_mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        collection_name=args.collection_name,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model
    ))
    
    # Exit with appropriate code
    sys.exit(0 if results is not None else 1)


if __name__ == "__main__":
    main()

