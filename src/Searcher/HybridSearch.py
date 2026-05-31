from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "LandLawDocumentCleaned.parquet"
GRAPH_NODES_PATH = PROJECT_ROOT / "data" / "legal_nodes.parquet"
GRAPH_EDGES_PATH = PROJECT_ROOT / "data" / "legal_edges.parquet"
GRAPH_CACHE_PATH = PROJECT_ROOT / "cache" / "legal_graph_adjacency.pkl"

GRAPH_RELATION_WEIGHTS = {
    "incoming": {
        "replaces": 0.95,
        "amends": 0.75,
        "guides": 0.45,
        "based_on": 0.25,
        "cites": 0.20,
    },
    "outgoing": {
        "replaces": 0.55,
        "amends": 0.45,
        "guides": 0.40,
        "based_on": 0.30,
        "cites": 0.22,
    },
}

REPLACED_RELATIONS = {"replaces", "amends"}

def minmax_normalize(results, score_key):
    results = [r for r in results if "chunk_id" in r]
    if not results:
        return {}

    scores = [r.get(score_key, 0.0) for r in results]
    min_s, max_s = min(scores), max(scores)

    if max_s == min_s:
        return {r["chunk_id"]: 1.0 for r in results}

    return {
        r["chunk_id"]: (r.get(score_key, 0.0) - min_s) / (max_s - min_s)
        for r in results
    }


SearchFn = Callable[..., List[Dict[str, Any]]]


def safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return [value]


def get_document_id(result: Dict[str, Any]) -> Optional[str]:
    document_id = result.get("document_id") or result.get("doc_id")
    if document_id is not None and str(document_id):
        return str(document_id)

    chunk_id = result.get("chunk_id")
    if chunk_id is None:
        return None
    return str(chunk_id).split("_", 1)[0]


class LegalGraphEnhancer:
    """Document-level citation graph expansion for hybrid search candidates."""

    def __init__(
        self,
        nodes_path: Path = GRAPH_NODES_PATH,
        edges_path: Path = GRAPH_EDGES_PATH,
        chunks_path: Path = DATA_PATH,
        cache_path: Path = GRAPH_CACHE_PATH,
    ):
        self.nodes_path = nodes_path
        self.edges_path = edges_path
        self.chunks_path = chunks_path
        self.cache_path = cache_path

        cached = self._load_adjacency_cache()
        if cached is not None:
            self.incoming_edges = cached["incoming_edges"]
            self.outgoing_edges = cached["outgoing_edges"]
        else:
            self.incoming_edges, self.outgoing_edges = self._build_adjacency()
            self._write_adjacency_cache()

    def _graph_metadata_path(self) -> Path:
        return self.cache_path.with_suffix(self.cache_path.suffix + ".meta.json")

    def _current_graph_metadata(self) -> Dict[str, Any]:
        edges_stat = self.edges_path.stat()
        return {
            "edges_path": str(self.edges_path.resolve()),
            "edges_size": edges_stat.st_size,
            "edges_mtime_ns": edges_stat.st_mtime_ns,
        }

    def _load_adjacency_cache(self) -> Optional[Dict[str, Any]]:
        metadata_path = self._graph_metadata_path()
        if not self.cache_path.exists() or not metadata_path.exists():
            return None

        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata != self._current_graph_metadata():
                return None

            print(f"Loading legal graph adjacency from cache: {self.cache_path}")
            start = time.time()
            with open(self.cache_path, "rb") as f:
                cached = pickle.load(f)
            print("Legal graph cache load time:", round(time.time() - start, 2), "seconds")
            return cached
        except Exception:
            return None

    def _write_adjacency_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_cache_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp_cache_path, "wb") as f:
            pickle.dump(
                {
                    "incoming_edges": self.incoming_edges,
                    "outgoing_edges": self.outgoing_edges,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_cache_path.replace(self.cache_path)

        tmp_metadata_path = self._graph_metadata_path().with_suffix(".json.tmp")
        with open(tmp_metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._current_graph_metadata(), f, ensure_ascii=False, indent=2)
        tmp_metadata_path.replace(self._graph_metadata_path())

    def _build_adjacency(self) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
        print(f"Building legal graph adjacency from: {self.edges_path}")
        start = time.time()
        edges = pd.read_parquet(self.edges_path)
        incoming_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        outgoing_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for edge in edges.to_dict(orient="records"):
            source_doc_id = edge.get("source_doc_id")
            target_doc_id = edge.get("target_doc_id")
            if source_doc_id is not None:
                outgoing_edges[str(source_doc_id)].append(edge)
            if target_doc_id is not None:
                incoming_edges[str(target_doc_id)].append(edge)

        print("Legal graph adjacency build time:", round(time.time() - start, 2), "seconds")
        return dict(incoming_edges), dict(outgoing_edges)

    @staticmethod
    def _chunk_sort_key(chunk_id: Any) -> tuple:
        chunk_id = "" if chunk_id is None else str(chunk_id)
        prefix, _, suffix = chunk_id.partition("_")
        try:
            suffix_number = int(suffix)
        except ValueError:
            suffix_number = 0
        return prefix, suffix_number

    def best_chunks_for_docs(
        self,
        doc_ids: List[str],
        preferred_chunk_ids_by_doc: Dict[str, List[Any]],
    ) -> Dict[str, Dict[str, Any]]:
        if not doc_ids:
            return {}

        columns = [
            "document_id", "document_number", "title", "legal_type",
            "issue_date", "effective_date", "source_url", "article",
            "clause", "point", "content", "references", "chunk_type",
            "chunk_id", "content_hash",
        ]
        chunks = pd.read_parquet(
            self.chunks_path,
            columns=columns,
            filters=[("document_id", "in", [str(doc_id) for doc_id in doc_ids])],
        )
        chunks = chunks[chunks["document_id"].astype(str).isin(set(doc_ids))]
        if chunks.empty:
            return {}

        rows_by_chunk_id = {
            str(row["chunk_id"]): row
            for row in chunks.to_dict(orient="records")
        }
        rows_by_doc_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in chunks.to_dict(orient="records"):
            rows_by_doc_id[str(row.get("document_id"))].append(row)

        best_by_doc_id = {}
        for doc_id in doc_ids:
            for chunk_id in preferred_chunk_ids_by_doc.get(doc_id, []):
                row = rows_by_chunk_id.get(str(chunk_id))
                if row is not None:
                    best_by_doc_id[doc_id] = deepcopy(row)
                    break
            if doc_id in best_by_doc_id:
                continue

            rows = rows_by_doc_id.get(doc_id, [])
            rows.sort(key=lambda row: self._chunk_sort_key(row.get("chunk_id")))
            with_content = [row for row in rows if str(row.get("content") or "").strip()]
            if rows:
                best_by_doc_id[doc_id] = deepcopy((with_content or rows)[0])

        return best_by_doc_id

    def expand(
        self,
        results: List[Dict[str, Any]],
        graph_weight: float = 0.25,
        max_edges_per_doc: int = 12,
        min_confidence: float = 0.75,
    ) -> List[Dict[str, Any]]:
        by_chunk_id = {str(result["chunk_id"]): deepcopy(result) for result in results if result.get("chunk_id")}
        graph_doc_scores: Dict[str, float] = defaultdict(float)
        graph_doc_relations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        penalties: Dict[str, float] = defaultdict(float)
        warnings: Dict[str, List[str]] = defaultdict(list)

        seed_docs = []
        for result in results:
            doc_id = get_document_id(result)
            if doc_id is not None:
                seed_docs.append((doc_id, float(result.get("final_score", result.get("score", 0.0)))))

        for seed_doc_id, seed_score in seed_docs:
            outgoing = self.outgoing_edges.get(seed_doc_id, [])[:max_edges_per_doc]
            incoming = self.incoming_edges.get(seed_doc_id, [])[:max_edges_per_doc]

            for edge in outgoing:
                self._apply_edge(
                    edge=edge,
                    direction="outgoing",
                    seed_doc_id=seed_doc_id,
                    seed_score=seed_score,
                    target_doc_id=str(edge.get("target_doc_id")),
                    graph_doc_scores=graph_doc_scores,
                    graph_doc_relations=graph_doc_relations,
                    min_confidence=min_confidence,
                )

            for edge in incoming:
                relation_type = str(edge.get("relation_type") or "")
                source_doc_id = str(edge.get("source_doc_id"))
                self._apply_edge(
                    edge=edge,
                    direction="incoming",
                    seed_doc_id=seed_doc_id,
                    seed_score=seed_score,
                    target_doc_id=source_doc_id,
                    graph_doc_scores=graph_doc_scores,
                    graph_doc_relations=graph_doc_relations,
                    min_confidence=min_confidence,
                )
                if relation_type in REPLACED_RELATIONS:
                    penalty = 0.18 if relation_type == "replaces" else 0.10
                    penalties[seed_doc_id] = max(penalties[seed_doc_id], penalty)
                    warnings[seed_doc_id].append(
                        f"Document has incoming '{relation_type}' edge from {source_doc_id}."
                    )

        for chunk_id, item in by_chunk_id.items():
            doc_id = get_document_id(item)
            if doc_id is None:
                continue

            graph_score = min(graph_doc_scores.get(doc_id, 0.0), 1.0)
            graph_penalty = penalties.get(doc_id, 0.0)
            item["graph_score"] = graph_score
            item["graph_penalty"] = graph_penalty
            item["graph_relations"] = graph_doc_relations.get(doc_id, [])
            if warnings.get(doc_id):
                item["legal_update_warning"] = warnings[doc_id]
            item["final_score"] = max(0.0, float(item.get("final_score", item.get("score", 0.0))) + graph_weight * graph_score - graph_penalty)
            item["score"] = item["final_score"]

        missing_doc_ids = []
        preferred_chunk_ids_by_doc: Dict[str, List[Any]] = defaultdict(list)
        for doc_id, graph_score in graph_doc_scores.items():
            if any(get_document_id(item) == doc_id for item in by_chunk_id.values()):
                continue

            relations = graph_doc_relations.get(doc_id, [])
            for relation in relations:
                if relation.get("direction") == "incoming":
                    preferred_chunk_ids_by_doc[doc_id].extend(safe_list(relation.get("source_chunk_id")))

            missing_doc_ids.append(doc_id)

        best_chunks = self.best_chunks_for_docs(missing_doc_ids, preferred_chunk_ids_by_doc)
        for doc_id in missing_doc_ids:
            graph_score = graph_doc_scores[doc_id]
            relations = graph_doc_relations.get(doc_id, [])
            item = best_chunks.get(doc_id)
            if item is None:
                continue

            chunk_id = str(item.get("chunk_id"))
            item["rank"] = None
            item["bm25_score"] = 0.0
            item["vector_score"] = 0.0
            item["graph_score"] = min(graph_score, 1.0)
            item["graph_penalty"] = 0.0
            item["graph_relations"] = relations
            item["final_score"] = graph_weight * item["graph_score"]
            item["score"] = item["final_score"]
            item["source"] = "hybrid_graph"
            by_chunk_id[chunk_id] = item

        merged = list(by_chunk_id.values())
        merged.sort(key=lambda item: item.get("final_score", item.get("score", 0.0)), reverse=True)
        return merged

    def _apply_edge(
        self,
        edge: Dict[str, Any],
        direction: str,
        seed_doc_id: str,
        seed_score: float,
        target_doc_id: str,
        graph_doc_scores: Dict[str, float],
        graph_doc_relations: Dict[str, List[Dict[str, Any]]],
        min_confidence: float,
    ) -> None:
        relation_type = str(edge.get("relation_type") or "")
        confidence = float(edge.get("confidence") or 0.0)
        if confidence < min_confidence or not target_doc_id:
            return

        relation_weight = GRAPH_RELATION_WEIGHTS.get(direction, {}).get(relation_type, 0.15)
        score = seed_score * confidence * relation_weight
        graph_doc_scores[target_doc_id] += score

        graph_doc_relations[target_doc_id].append({
            "seed_doc_id": seed_doc_id,
            "direction": direction,
            "relation_type": relation_type,
            "confidence": confidence,
            "graph_score": score,
            "source_doc_id": edge.get("source_doc_id"),
            "target_doc_id": edge.get("target_doc_id"),
            "source_chunk_id": edge.get("source_chunk_id"),
            "citation_text": edge.get("citation_text"),
            "evidence_text": edge.get("evidence_text"),
        })


_default_graph_enhancer: Optional[LegalGraphEnhancer] = None


def get_default_graph_enhancer() -> LegalGraphEnhancer:
    global _default_graph_enhancer
    if _default_graph_enhancer is None:
        _default_graph_enhancer = LegalGraphEnhancer()
    return _default_graph_enhancer


def _default_bm25_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    try:
        from .BM25Search import bm25_search
    except ImportError:
        from BM25Search import bm25_search
    return bm25_search(query, top_k=top_k)


def _default_vector_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    try:
        from .VectorSearch import vector_search
    except ImportError:
        from VectorSearch import vector_search
    return vector_search(query, top_k=top_k)


def hybrid_search(
    query: str,
    bm25_search_fn: Optional[SearchFn] = None,
    vector_search_fn: Optional[SearchFn] = None,
    top_k: int = 20,
    pool_k: int = 100,
    alpha: float = 0.4,
    use_graph: bool = True,
    graph_weight: float = 0.25,
    graph_expansion_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    bm25_search_fn = bm25_search_fn or _default_bm25_search
    vector_search_fn = vector_search_fn or _default_vector_search

    bm25_results = bm25_search_fn(query, top_k=pool_k)
    vector_results = vector_search_fn(query, top_k=pool_k)

    bm25_scores = minmax_normalize(bm25_results, "score")
    vector_scores = minmax_normalize(vector_results, "score")

    all_chunk_ids = set(bm25_scores) | set(vector_scores)
    by_chunk_id: Dict[str, Dict[str, Any]] = {}

    for result in bm25_results + vector_results:
        chunk_id = result.get("chunk_id")
        if chunk_id is not None and chunk_id not in by_chunk_id:
            by_chunk_id[chunk_id] = result

    merged = []

    for chunk_id in all_chunk_ids:
        bm25_score = bm25_scores.get(chunk_id, 0.0)
        vector_score = vector_scores.get(chunk_id, 0.0)

        final_score = (1 - alpha) * bm25_score + alpha * vector_score

        item = deepcopy(by_chunk_id.get(chunk_id, {"chunk_id": chunk_id}))
        item["bm25_score"] = bm25_score
        item["vector_score"] = vector_score
        item["final_score"] = final_score
        item["score"] = final_score
        item["source"] = "hybrid"
        merged.append(item)

    merged.sort(key=lambda x: x["final_score"], reverse=True)
    if use_graph:
        expansion_pool = merged[: graph_expansion_k or pool_k]
        graph_enhancer = get_default_graph_enhancer()
        merged = graph_enhancer.expand(expansion_pool, graph_weight=graph_weight)

    for rank, item in enumerate(merged[:top_k], start=1):
        item["rank"] = rank

    return merged[:top_k]


if __name__ == "__main__":
    print("HybridSearch is ready. Import hybrid_search(query) to run retrieval.")
