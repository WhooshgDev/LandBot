import json
from pathlib import Path
from typing import List, Dict, Any


# =========================================================
# Load evaluation queries
# =========================================================

def load_eval_queries(path: str | Path) -> List[Dict[str, Any]]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


# =========================================================
# Metrics
# =========================================================

def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """
    Recall@k = 1 if at least one gold chunk appears in top-k, else 0.

    For your synthetic benchmark, each query usually has 1 gold chunk.
    """
    if not gold_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)

    return 1.0 if len(top_k & gold_set) > 0 else 0.0


def reciprocal_rank_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """
    MRR@k contribution for one query.

    If the first relevant result appears at rank r:
        reciprocal rank = 1 / r

    If no relevant result appears in top-k:
        reciprocal rank = 0
    """
    if not gold_ids:
        return 0.0

    gold_set = set(gold_ids)

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in gold_set:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    eval_queries: List[Dict[str, Any]],
    retrieval_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, float]:
    """
    retrieval_results format:

    {
        "q000001": [
            {"chunk_id": "chunk_123", "score": 0.91},
            {"chunk_id": "chunk_456", "score": 0.87}
        ],
        "q000002": [...]
    }
    """

    recall_5_scores = []
    recall_10_scores = []
    mrr_10_scores = []

    missing_results = 0

    for item in eval_queries:
        query_id = item["query_id"]
        gold_ids = item.get("gold_chunk_ids", [])

        results = retrieval_results.get(query_id, [])

        if not results:
            missing_results += 1

        retrieved_ids = [str(r["chunk_id"]) for r in results]

        recall_5_scores.append(recall_at_k(retrieved_ids, gold_ids, k=5))
        recall_10_scores.append(recall_at_k(retrieved_ids, gold_ids, k=10))
        mrr_10_scores.append(reciprocal_rank_at_k(retrieved_ids, gold_ids, k=10))

    n = len(eval_queries)

    return {
        "num_queries": n,
        "missing_results": missing_results,
        "recall@5": sum(recall_5_scores) / n if n else 0.0,
        "recall@10": sum(recall_10_scores) / n if n else 0.0,
        "mrr@10": sum(mrr_10_scores) / n if n else 0.0,
    }


# =========================================================
# Example usage
# =========================================================

if __name__ == "__main__":
    eval_path = Path("data/eval_queries.jsonl")
    eval_queries = load_eval_queries(eval_path)

    # -----------------------------------------------------
    # Replace this part with your actual retrieval function.
    # -----------------------------------------------------
    retrieval_results = {}

    for item in eval_queries:
        query_id = item["query_id"]
        query = item["query"]

        # Example placeholder:
        # results = your_retriever.search(query, top_k=10)

        results = [
            # {"chunk_id": "chunk_123", "score": 0.91},
            # {"chunk_id": "chunk_456", "score": 0.87},
        ]

        retrieval_results[query_id] = results

    metrics = evaluate_retrieval(eval_queries, retrieval_results)

    print("\nEvaluation Results")
    print("==================")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")