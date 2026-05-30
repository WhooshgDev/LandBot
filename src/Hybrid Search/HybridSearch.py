def minmax_normalize(results, score_key):
    if not results:
        return {}

    scores = [r[score_key] for r in results]
    min_s, max_s = min(scores), max(scores)

    if max_s == min_s:
        return {r["chunk_id"]: 1.0 for r in results}

    return {
        r["chunk_id"]: (r[score_key] - min_s) / (max_s - min_s)
        for r in results
    }


def hybrid_search(query, bm25_search, vector_search, top_k=20, pool_k=100, alpha=0.4):
    bm25_results = bm25_search(query, top_k=pool_k)
    vector_results = vector_search(query, top_k=pool_k)

    bm25_scores = minmax_normalize(bm25_results, "score")
    vector_scores = minmax_normalize(vector_results, "score")

    all_chunk_ids = set(bm25_scores) | set(vector_scores)

    merged = []

    for chunk_id in all_chunk_ids:
        bm25_score = bm25_scores.get(chunk_id, 0.0)
        vector_score = vector_scores.get(chunk_id, 0.0)

        final_score = (1 - alpha) * bm25_score + alpha * vector_score

        merged.append({
            "chunk_id": chunk_id,
            "bm25_score": bm25_score,
            "vector_score": vector_score,
            "final_score": final_score,
        })

    merged.sort(key=lambda x: x["final_score"], reverse=True)

    return merged[:top_k]