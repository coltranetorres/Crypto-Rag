"""
RAGAS Evaluation Module
Integrates RAGAS with existing RAG system for evaluation
"""
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from ragas.metrics import DiscreteMetric
from ragas import evaluate
from ragas.llms import llm_factory
from datasets import Dataset
from src.rag_agent import RAGAgent
import mlflow
from datetime import datetime
import os
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def create_correctness_metric():
    """
    Create correctness metric instance (LLM is passed to score method, not constructor)
    
    Returns:
        Configured DiscreteMetric instance
    """
    return DiscreteMetric(
        name="correctness",
        prompt="""Compare the model response to the expected answer and determine if it's correct.

Consider the response correct if it:
1. Contains the key information from the expected answer
2. Is factually accurate based on the provided context
3. Adequately addresses the question asked

Return 'pass' if the response is correct, 'fail' if it's incorrect.

Question: {question}
Expected Answer: {expected_answer}
Model Response: {response}

Evaluation:""",
        allowed_values=["pass", "fail"],
    )


def create_ragas_llm(openrouter_provider, llm=None):
    """
    Create RAGAS LLM instance from OpenRouter provider
    
    Args:
        openrouter_provider: OpenRouterProvider instance
        llm: Optional existing RAGAS LLM instance
    
    Returns:
        RAGAS LLM instance
    """
    if llm is not None:
        return llm
    
    if openrouter_provider is None:
        raise ValueError("openrouter_provider must be provided if llm is None")
    
    # Use the OpenAI client from OpenRouterProvider (OpenRouter uses OpenAI-compatible API)
    # Use the provider's configured model - OpenRouter models work directly with OpenAI client
    model_name = openrouter_provider.default_llm_model
    # Try with model name as-is first (OpenRouter models work directly)
    try:
        llm = llm_factory(model_name, client=openrouter_provider.client)
    except Exception:
        # Fallback: some RAGAS versions might need "openai/" prefix for OpenAI-compatible APIs
        ragas_model_name = f"openai/{model_name}" if not model_name.startswith("openai/") else model_name
        llm = llm_factory(ragas_model_name, client=openrouter_provider.client)
    
    return llm


def load_qa_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load ground truth dataset from CSV file
    
    Args:
        csv_path: Path to qa_data.csv
        
    Returns:
        DataFrame with columns: ID, Document, Query, Page Number, Answer
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} Q&A pairs from {csv_path}")
    return df


def create_ragas_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert DataFrame to RAGAS evaluation dataset format
    
    Args:
        df: DataFrame with Query and Answer columns
        
    Returns:
        RAGAS Dataset with columns: question, ground_truth
    """
    # Create dataset in RAGAS format
    dataset_dict = {
        "question": df["Query"].tolist(),
        "ground_truth": df["Answer"].tolist(),
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Created RAGAS dataset with {len(dataset)} samples")
    return dataset


def run_rag_query(rag_agent: RAGAgent, question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Run a query through the RAG system and format for RAGAS evaluation
    
    Args:
        rag_agent: Initialized RAGAgent instance
        question: User question
        top_k: Number of documents to retrieve
        
    Returns:
        Dictionary with answer and contexts formatted for RAGAS
    """
    result = rag_agent.query_direct(question, top_k=top_k)
    
    # Format contexts for RAGAS (list of strings)
    contexts = [ctx["text"] for ctx in result.get("contexts", [])]
    
    return {
        "answer": result["answer"],
        "contexts": contexts,
        "question": question,
        "metadata": result.get("metadata", {})
    }


async def evaluate_rag(
    rag_agent: RAGAgent,
    dataset: Dataset,
    experiment_name: Optional[str] = None,
    top_k: int = 5,
    openrouter_provider=None,
    llm=None,
    use_mlflow: bool = True,
    mlflow_tracking_uri: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate RAG system using RAGAS with optional MLflow tracing
    
    Args:
        rag_agent: Initialized RAGAgent instance
        dataset: RAGAS Dataset with question and ground_truth columns
        experiment_name: Optional name for this experiment
        top_k: Number of documents to retrieve per query
        openrouter_provider: Optional OpenRouterProvider instance (uses its OpenAI client for metric)
        llm: Optional RAGAS LLM instance for the metric (alternative to openrouter_provider)
        use_mlflow: Whether to use MLflow for tracing (default: True)
        mlflow_tracking_uri: Optional MLflow tracking URI (default: local file store)
        
    Returns:
        List of evaluation results with scores and metadata
    """
    logger.info(f"Starting evaluation on {len(dataset)} samples...")
    
    # Setup MLflow if enabled
    if use_mlflow:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        else:
            # Use local file store (default)
            mlflow.set_tracking_uri("file:./mlruns")
        
        # Set experiment name
        exp_name = experiment_name or f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.set_experiment(exp_name)
        
        # Start parent run
        with mlflow.start_run(run_name=exp_name) as parent_run:
            mlflow.log_param("num_samples", len(dataset))
            mlflow.log_param("top_k", top_k)
            mlflow.log_param("experiment_name", exp_name)
            
            # Create metric (LLM is passed to score method, not constructor)
            correctness_metric = create_correctness_metric()
            
            # Create RAGAS LLM from OpenRouter provider
            ragas_llm = create_ragas_llm(openrouter_provider, llm=llm)
            
            results = []
            
            for i, sample in enumerate(dataset):
                question = sample["question"]
                ground_truth = sample["ground_truth"]
                
                logger.info(f"[{i+1}/{len(dataset)}] Evaluating: {question[:60]}...")
                
                # Create nested run for each evaluation sample
                sample_run_name = f"sample_{i+1}"
                
                try:
                    with mlflow.start_run(run_name=sample_run_name, nested=True) as sample_run:
                        # Log sample parameters
                        mlflow.log_param("question", question)
                        mlflow.log_param("expected_answer", ground_truth)
                        mlflow.log_param("top_k", top_k)
                        
                        # Run RAG query
                        rag_result = run_rag_query(rag_agent, question, top_k=top_k)
                        
                        # Log retrieval metrics
                        mlflow.log_metric("contexts_retrieved", len(rag_result["contexts"]))
                        mlflow.log_metric("retrieval_latency_ms", rag_result.get("metadata", {}).get("retrieval_latency_ms", 0))
                        mlflow.log_metric("generation_latency_ms", rag_result.get("metadata", {}).get("generation_latency_ms", 0))
                        mlflow.log_metric("total_latency_ms", rag_result.get("metadata", {}).get("total_latency_ms", 0))
                        
                        # Evaluate with metric
                        score_result = correctness_metric.score(
                            question=question,
                            expected_answer=ground_truth,
                            response=rag_result["answer"],
                            llm=ragas_llm,
                        )
                        
                        # Extract score value
                        correctness_score = score_result.value if hasattr(score_result, 'value') else str(score_result)
                        
                        # Log correctness as metric (1.0 for pass, 0.0 for fail)
                        correctness_numeric = 1.0 if correctness_score == "pass" else 0.0
                        mlflow.log_metric("correctness", correctness_numeric)
                        mlflow.log_param("correctness_score", correctness_score)
                        mlflow.log_param("model_response", rag_result["answer"][:500])  # Truncate long responses
                        
                        # Store result with MLflow info
                        result = {
                            "question": question,
                            "expected_answer": ground_truth,
                            "model_response": rag_result["answer"],
                            "correctness_score": correctness_score,
                            "contexts_retrieved": len(rag_result["contexts"]),
                            "metadata": rag_result["metadata"],
                            "mlflow_run_id": sample_run.info.run_id,
                            "mlflow_run_name": sample_run_name,
                        }
                        
                        results.append(result)
                        
                        logger.debug(f"Score: {correctness_score}")
                
                except Exception as e:
                    logger.error(f"Error evaluating sample: {e}", exc_info=True)
                    results.append({
                        "question": question,
                        "expected_answer": ground_truth,
                        "model_response": "",
                        "correctness_score": "error",
                        "error": str(e),
                    })
            
            # Log summary metrics
            if results:
                pass_count = sum(1 for r in results if r.get("correctness_score") == "pass")
                total_count = len(results)
                pass_rate = (pass_count / total_count) * 100 if total_count > 0 else 0
                
                mlflow.log_metric("pass_count", pass_count)
                mlflow.log_metric("total_count", total_count)
                mlflow.log_metric("pass_rate", pass_rate)
                
                logger.info(f"Evaluation complete: {len(results)} results")
                logger.info(f"Pass rate: {pass_count}/{total_count} ({pass_rate:.1f}%)")
            
            return results
    else:
        # No MLflow - simple evaluation
        correctness_metric = create_correctness_metric()
        ragas_llm = create_ragas_llm(openrouter_provider, llm=llm)
        
        results = []
        
        for i, sample in enumerate(dataset):
            question = sample["question"]
            ground_truth = sample["ground_truth"]
            
            print(f"[{i+1}/{len(dataset)}] Evaluating: {question[:60]}...")
            
            try:
                rag_result = run_rag_query(rag_agent, question, top_k=top_k)
                
                score_result = correctness_metric.score(
                    question=question,
                    expected_answer=ground_truth,
                    response=rag_result["answer"],
                    llm=ragas_llm,
                )
                
                correctness_score = score_result.value if hasattr(score_result, 'value') else str(score_result)
                
                result = {
                    "question": question,
                    "expected_answer": ground_truth,
                    "model_response": rag_result["answer"],
                    "correctness_score": correctness_score,
                    "contexts_retrieved": len(rag_result["contexts"]),
                    "metadata": rag_result["metadata"],
                }
                
                results.append(result)
                print(f"  [OK] Score: {correctness_score}")
                
            except Exception as e:
                print(f"  [ERROR] Error: {e}")
                results.append({
                    "question": question,
                    "expected_answer": ground_truth,
                    "model_response": "",
                    "correctness_score": "error",
                    "error": str(e),
                })
        
        logger.info(f"Evaluation complete: {len(results)} results")
        return results


def save_results(results: List[Dict[str, Any]], output_dir: str = "experiments", experiment_name: Optional[str] = None):
    """
    Save evaluation results to CSV
    
    Args:
        results: List of evaluation result dictionaries
        output_dir: Directory to save results
        experiment_name: Optional experiment name for filename
    """
    import os
    from datetime import datetime
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    if experiment_name:
        filename = f"{experiment_name}_results.csv"
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"evaluation_{timestamp}_results.csv"
    
    filepath = os.path.join(output_dir, filename)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    
    logger.info(f"Results saved to: {filepath}")
    
    # Print summary
    if results:
        pass_count = sum(1 for r in results if r.get("correctness_score") == "pass")
        total_count = len(results)
        pass_rate = (pass_count / total_count) * 100 if total_count > 0 else 0
        logger.info(f"Summary - Pass rate: {pass_count}/{total_count} ({pass_rate:.1f}%)")
    
    return filepath

