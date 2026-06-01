from __future__ import annotations

import argparse
import contextlib
import io
import time
from typing import Any, Dict, Sequence

from Searcher.HybridSearch import hybrid_search


TRIAL_QUERY = "Quy định về thu hồi đất và xử lý quyền lợi, nghĩa vụ của người sử dụng đất khi Nhà nước thu hồi đất là gì?"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def print_results(results: Sequence[Dict[str, Any]]) -> None:
    print("Sources")
    print("=" * 100)
    for result in results:
        print("-" * 100)
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Content: {clean_text(result.get('content'))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LandBot retrieval.")
    parser.add_argument("--query", default=TRIAL_QUERY, help="Question to answer.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieval sources.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Hybrid vector weight.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF rank smoothing constant.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        results = hybrid_search(args.query, top_k=args.top_k, alpha=args.alpha, rrf_k=args.rrf_k)
    loading_time = time.time() - start

    print(f"Loading time: {loading_time:.2f} seconds")
    print()
    print_results(results)


if __name__ == "__main__":
    main()
