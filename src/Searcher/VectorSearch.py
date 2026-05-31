from __future__ import annotations

import json
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
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
        self.tokenizer = None
        self.model = None

    def _validate_store(self) -> None:
        if self.index.ntotal != self.config["num_vectors"]:
            raise ValueError("Vector count mismatch!")
        if self.index.d != self.config["dim"]:
            raise ValueError("Dimension mismatch!")
        if len(self.chunk_ids) != len(self.metadata):
            raise ValueError("Metadata length mismatch!")

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


if __name__ == "__main__":
    retriever = get_default_retriever()
    print(f"Index size: {retriever.index.ntotal} vectors, dimension: {retriever.index.d}")
    print(f"Metadata: {len(retriever.chunk_ids)} entries")
