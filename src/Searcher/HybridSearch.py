from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple


SearchFn = Callable[..., List[Dict[str, Any]]]


def retrieve_candidates(
    query: str,
    pool_k: int = 100,
    bm25_search_fn: Optional[SearchFn] = None,
    vector_search_fn: Optional[SearchFn] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if bm25_search_fn is None:
        try:
            from .BM25Search import bm25_search as bm25_search_fn
        except ImportError:
            from BM25Search import bm25_search as bm25_search_fn

    if vector_search_fn is None:
        try:
            from .VectorSearch import vector_search as vector_search_fn
        except ImportError:
            from VectorSearch import vector_search as vector_search_fn

    return (
        bm25_search_fn(query, top_k=pool_k),
        vector_search_fn(query, top_k=pool_k),
    )


def rank_by_chunk_id(results: List[Dict[str, Any]]) -> Dict[str, int]:
    ranks = {}
    for fallback_rank, result in enumerate(results, start=1):
        chunk_id = result.get("chunk_id")
        if chunk_id is not None:
            ranks[str(chunk_id)] = int(result.get("rank") or fallback_rank)
    return ranks


def score_by_chunk_id(
    results: List[Dict[str, Any]],
    preferred_score_key: str,
) -> Dict[str, float]:
    scores = {}
    for result in results:
        chunk_id = result.get("chunk_id")
        if chunk_id is None:
            continue

        # Prefer the retriever-specific raw score, but fall back to "score".
        score = result.get(preferred_score_key, result.get("score", 0.0))
        scores[str(chunk_id)] = float(score or 0.0)

    return scores


def rrf_fuse(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    top_k: int = 10,
    alpha: float = 0.6,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    RRF fusion.

    alpha controls vector weight:
        alpha = 0.5 means BM25 and vector are equal
        alpha = 0.6 means vector is slightly stronger
        alpha = 0.4 means BM25 is stronger
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")

    bm25_weight = 1.0 - alpha
    vector_weight = alpha

    bm25_ranks = rank_by_chunk_id(bm25_results)
    vector_ranks = rank_by_chunk_id(vector_results)

    bm25_raw_scores = score_by_chunk_id(bm25_results, "bm25_score")
    vector_raw_scores = score_by_chunk_id(vector_results, "vector_score")

    all_chunk_ids = set(bm25_ranks) | set(vector_ranks)

    by_chunk_id: Dict[str, Dict[str, Any]] = {}
    for result in bm25_results + vector_results:
        chunk_id = result.get("chunk_id")
        if chunk_id is not None:
            by_chunk_id.setdefault(str(chunk_id), result)

    fused = []

    for chunk_id in all_chunk_ids:
        bm25_rank = bm25_ranks.get(chunk_id)
        vector_rank = vector_ranks.get(chunk_id)

        bm25_rrf_score = (
            bm25_weight / (rrf_k + bm25_rank)
            if bm25_rank is not None
            else 0.0
        )

        vector_rrf_score = (
            vector_weight / (rrf_k + vector_rank)
            if vector_rank is not None
            else 0.0
        )

        fusion_score = bm25_rrf_score + vector_rrf_score

        item = deepcopy(by_chunk_id.get(chunk_id, {"chunk_id": chunk_id}))

        # Raw retriever scores
        item["bm25_raw_score"] = bm25_raw_scores.get(chunk_id)
        item["vector_raw_score"] = vector_raw_scores.get(chunk_id)

        # Ranks
        item["bm25_rank"] = bm25_rank
        item["vector_rank"] = vector_rank

        # RRF contribution scores
        item["bm25_rrf_score"] = bm25_rrf_score
        item["vector_rrf_score"] = vector_rrf_score

        # Final fused score
        item["fusion"] = "rrf"
        item["fusion_score"] = fusion_score
        item["final_score"] = fusion_score
        item["score"] = fusion_score
        item["source"] = "hybrid"

        fused.append(item)

    fused.sort(key=lambda item: item["final_score"], reverse=True)

    for rank, item in enumerate(fused[:top_k], start=1):
        item["rank"] = rank

    return fused[:top_k]


def hybrid_search(
    query: str,
    top_k: int = 10,
    pool_k: int = 100,
    bm25_search_fn: Optional[SearchFn] = None,
    vector_search_fn: Optional[SearchFn] = None,
    alpha: float = 0.6,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    bm25_results, vector_results = retrieve_candidates(
        query=query,
        pool_k=pool_k,
        bm25_search_fn=bm25_search_fn,
        vector_search_fn=vector_search_fn,
    )

    return rrf_fuse(
        bm25_results=bm25_results,
        vector_results=vector_results,
        top_k=top_k,
        alpha=alpha,
        rrf_k=rrf_k,
    )


if __name__ == "__main__":
    print("HybridSearch is ready. Import hybrid_search(query) to run RRF retrieval.")