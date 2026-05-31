from __future__ import annotations

import time
from typing import Any, Dict, Sequence

from Searcher.HybridSearch import hybrid_search


TRIAL_QUERY = "Quy định về thu hồi đất và xử lý quyền lợi, nghĩa vụ của người sử dụng đất khi Nhà nước thu hồi đất là gì?"


def safe_preview(value: Any, limit: int = 260) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def format_score(value, digits=4):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def print_results(results: Sequence[Dict[str, Any]]) -> None:
    for result in results:
        print("-" * 100)
        print(f"Rank: {result.get('rank')}")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Fused score: {format_score(result.get('score', 0.0))}")
        print(f"BM25 raw: {format_score(result.get('bm25_raw_score'))}")
        print(f"BM25 rank: {result.get('bm25_rank')}")
        print(f"Vector raw: {format_score(result.get('vector_raw_score'))}")
        print(f"Vector rank: {result.get('vector_rank')}")
        print(f"RRF: {format_score(result.get('fusion_score', result.get('final_score', 0.0)))}")        
        print(f"Title: {safe_preview(result.get('title'), 180)}")
        print(f"Content: {safe_preview(result.get('content'))}")


def main() -> None:
    print("Trial query:")
    print(TRIAL_QUERY)
    print()

    start = time.time()
    results = hybrid_search(TRIAL_QUERY, top_k=5, alpha=0.4)
    
    runtime = time.time() - start

    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Result count: {len(results)}")
    print_results(results)


if __name__ == "__main__":
    main()
