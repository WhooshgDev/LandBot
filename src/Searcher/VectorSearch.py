from __future__ import annotations

import json
import pickle
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
DATA_PATH = PROJECT_ROOT / "data" / "LandLawDocumentCleaned.parquet"
MODEL_NAME = "BAAI/bge-m3"
MAX_SEQ_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VectorRetriever:
    """FAISS vector retriever for legal chunks."""

    def __init__(
        self,
        vector_store_dir: Path = VECTOR_STORE_DIR,
        model_name: str = MODEL_NAME,
        max_seq_len: int = MAX_SEQ_LEN,
        device: Optional[str] = None,
    ):
        self.vector_store_dir = Path(vector_store_dir)
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.device = device or DEVICE

        with open(self.vector_store_dir / "config.json", encoding="utf-8") as f:
            self.config = json.load(f)

        self.index = faiss.read_index(str(self.vector_store_dir / "faiss_index.bin"))

        with open(self.vector_store_dir / "metadata.pkl", "rb") as f:
            data = pickle.load(f)
        self.chunk_ids = data["chunk_ids"]
        self.metadata = data["metadata"]

        self._validate_store()
        self.content_by_chunk_id: Dict[str, str] = {}
        self.tokenizer = None
        self.model = None

    def _validate_store(self) -> None:
        if self.index.ntotal != self.config["num_vectors"]:
            raise ValueError("Vector count mismatch!")
        if self.index.d != self.config["dim"]:
            raise ValueError("Dimension mismatch!")
        if len(self.chunk_ids) != len(self.metadata):
            raise ValueError("Metadata length mismatch!")
        if "embedding_text_version" not in self.config:
            print(
                "Warning: vector store config has no embedding_text_version. "
                "It was likely built from content-only embeddings; rebuild with src/Embedding/Embedder.py."
            )

    def _hydrate_missing_content(self, items: List[Dict[str, Any]]) -> None:
        missing_chunk_ids = [
            str(item["chunk_id"])
            for item in items
            if "content" not in item and str(item.get("chunk_id")) not in self.content_by_chunk_id
        ]
        if not missing_chunk_ids:
            return

        df = pd.read_parquet(
            DATA_PATH,
            columns=["chunk_id", "content"],
            filters=[("chunk_id", "in", missing_chunk_ids)],
        )
        self.content_by_chunk_id.update({
            str(row["chunk_id"]): "" if row["content"] is None else str(row["content"])
            for row in df.to_dict(orient="records")
        })

        for chunk_id in missing_chunk_ids:
            self.content_by_chunk_id.setdefault(chunk_id, "")

    def _load_model(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            self.model_name,
            attn_implementation="sdpa",
            dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text: str) -> np.ndarray:
        self._load_model()
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0]
            emb = F.normalize(emb, p=2, dim=1)

        return emb.cpu().numpy().astype(np.float32)

    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []

        emb = self.embed_text(query)
        scores, indices = self.index.search(emb, k=min(top_k, self.index.ntotal))

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue

            item = deepcopy(self.metadata[int(idx)])
            item.setdefault("chunk_id", self.chunk_ids[int(idx)])
            item["rank"] = rank
            item["vector_score"] = float(score)
            item["score"] = float(score)
            item["source"] = "vector"
            results.append(item)

        self._hydrate_missing_content(results)
        for item in results:
            item.setdefault("content", self.content_by_chunk_id.get(str(item.get("chunk_id")), ""))

        return results

    def __call__(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        return self.search(query, top_k=top_k)


_default_retriever: Optional[VectorRetriever] = None


def get_default_retriever() -> VectorRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = VectorRetriever()
    return _default_retriever


def vector_search(query: str, top_k: int = 100) -> List[Dict[str, Any]]:
    return get_default_retriever().search(query, top_k=top_k)


TRIAL_QUERY = "Điều kiện cấp giấy chứng nhận quyền sử dụng đất là gì?"


def safe_preview(value: Any, limit: int = 240) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def print_vector_results(results: Sequence[Dict[str, Any]]) -> None:
    for result in results:
        print("-" * 100)
        print(f"Rank: {result.get('rank')}")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print(f"Document ID: {result.get('document_id')}")
        print(f"Score: {result.get('score', 0.0):.6f}")
        print(f"Vector score: {result.get('vector_score', 0.0):.6f}")
        print(f"Document number: {safe_preview(result.get('document_number'), 120)}")
        print(f"Title: {safe_preview(result.get('title'), 180)}")
        print(f"Content: {safe_preview(result.get('content'))}")


if __name__ == "__main__":
    print("VectorSearch trial query:")
    print(TRIAL_QUERY)
    print()

    start = time.time()
    retriever = get_default_retriever()
    print(f"Retriever load time: {time.time() - start:.2f} seconds")
    print(f"Index size: {retriever.index.ntotal} vectors, dimension: {retriever.index.d}")
    print(f"Metadata: {len(retriever.chunk_ids)} entries")
    print(f"FAISS index type: {type(retriever.index).__name__}")
    print(f"Configured model: {retriever.model_name}")
    print(f"Device: {retriever.device}")
    print()

    start = time.time()
    query_embedding = retriever.embed_text(TRIAL_QUERY)
    print(f"Embedding time: {time.time() - start:.2f} seconds")
    print(f"Embedding shape: {query_embedding.shape}")
    print(f"Embedding L2 norm: {float(np.linalg.norm(query_embedding)):.6f}")
    print()

    start = time.time()
    scores, indices = retriever.index.search(query_embedding, k=5)
    print(f"Raw FAISS search time: {time.time() - start:.4f} seconds")
    print("Raw FAISS indices:", indices[0].tolist())
    print("Raw FAISS scores:", [round(float(score), 6) for score in scores[0]])
    print()

    start = time.time()
    results = retriever.search(TRIAL_QUERY, top_k=5)
    print(f"Full vector_search time: {time.time() - start:.4f} seconds")
    print(f"Result count: {len(results)}")
    print_vector_results(results)
