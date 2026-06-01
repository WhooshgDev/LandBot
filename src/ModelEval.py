from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path
from typing import Any, Dict, List

from Evaluation.EvalMetrics import evaluate_retrieval, load_eval_queries
from Searcher.HybridSearch import hybrid_search


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval_queries.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LandBot retrieval metrics.")
    parser.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH), help="Path to eval_queries.jsonl.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of retrieved chunks per query.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Hybrid vector weight.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF rank smoothing constant.")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print progress while evaluating queries.",
    )
    return parser.parse_args()


def retrieve_for_eval_query(query: str, top_k: int, alpha: float, rrf_k: int) -> List[Dict[str, Any]]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return hybrid_search(query, top_k=top_k, alpha=alpha, rrf_k=rrf_k)


def run_evaluation(args: argparse.Namespace) -> Dict[str, float]:
    eval_queries = load_eval_queries(args.eval_path)
    retrieval_results: Dict[str, List[Dict[str, Any]]] = {}

    for index, item in enumerate(eval_queries, start=1):
        query_id = item["query_id"]
        query = item["query"]

        if args.show_progress:
            print(f"Evaluating {index}/{len(eval_queries)}: {query_id}")

        retrieval_results[query_id] = retrieve_for_eval_query(
            query=query,
            top_k=args.top_k,
            alpha=args.alpha,
            rrf_k=args.rrf_k,
        )

    return evaluate_retrieval(eval_queries, retrieval_results)


def print_metrics(metrics: Dict[str, float], elapsed_time: float) -> None:
    print("Evaluation Results")
    print("=" * 100)
    print(f"num_queries: {int(metrics['num_queries'])}")
    print(f"missing_results: {int(metrics['missing_results'])}")
    print(f"recall@5: {metrics['recall@5']:.4f}")
    print(f"recall@10: {metrics['recall@10']:.4f}")
    print(f"mrr@10: {metrics['mrr@10']:.4f}")
    print(f"evaluation_time: {elapsed_time:.2f} seconds")


def main() -> None:
    args = parse_args()

    start = time.time()
    metrics = run_evaluation(args)
    elapsed_time = time.time() - start

    print_metrics(metrics, elapsed_time)


if __name__ == "__main__":
    main()
