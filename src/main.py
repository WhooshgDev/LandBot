from __future__ import annotations

import time
from typing import Any, Dict, Sequence

from Searcher.HybridSearch import hybrid_search


TRIAL_QUERY = "Điều kiện cấp giấy chứng nhận quyền sử dụng đất là gì?"


def safe_preview(value: Any, limit: int = 260) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_results(results: Sequence[Dict[str, Any]]) -> None:
    for result in results:
        print("-" * 100)
        print(f"Rank: {result.get('rank')}")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Score: {result.get('score', 0.0):.4f}")
        print(f"BM25: {result.get('bm25_score', 0.0):.4f}")
        print(f"Vector: {result.get('vector_score', 0.0):.4f}")
        print(f"Graph: {result.get('graph_score', 0.0):.4f}")
        if result.get("graph_penalty"):
            print(f"Graph penalty: {result.get('graph_penalty', 0.0):.4f}")
        if result.get("legal_update_warning"):
            print(f"Legal update warning: {result.get('legal_update_warning')}")
        if result.get("graph_relations"):
            relation_preview = [
                f"{rel.get('direction')}:{rel.get('relation_type')}:{rel.get('seed_doc_id')}"
                for rel in result.get("graph_relations", [])[:3]
            ]
            print(f"Graph relations: {relation_preview}")
        print(f"Title: {safe_preview(result.get('title'), 180)}")
        print(f"Content: {safe_preview(result.get('content'))}")


def main() -> None:
    print("Trial query:")
    print(TRIAL_QUERY)
    print()

    start = time.time()
    results = hybrid_search(TRIAL_QUERY, top_k=5, pool_k=20, alpha=0.4)
    runtime = time.time() - start

    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Result count: {len(results)}")
    print_results(results)


if __name__ == "__main__":
    main()
