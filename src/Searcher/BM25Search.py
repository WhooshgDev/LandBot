from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json
import os
import pickle
import re
import time
import unicodedata

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except Exception:
    ViTokenizer = None
    PYVI_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception as exc:
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False
    print("sentence-transformers/CrossEncoder chưa import được:", repr(exc))

print("PyVi available:", PYVI_AVAILABLE)
print("CrossEncoder available:", CROSS_ENCODER_AVAILABLE)

# =========================
# Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "LandLawDocumentCleaned.parquet"  # đổi nếu file nằm nơi khác
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Với 300k chunks, lần đầu build BM25 có thể lâu.
# Dùng cache để lần sau không tokenize lại.
USE_TOKEN_CACHE = True
TOKENIZER_NAME = "pyvi"  # options: "pyvi", "simple"
TOKEN_CACHE_PATH = CACHE_DIR / f"bm25_tokenized_{TOKENIZER_NAME}.pkl"

# Nên lấy candidate rộng hơn top-10 để reranker có cơ hội tìm đúng hơn.
BM25_TOP_N = 100
FINAL_TOP_K = 10

# Query gốc dùng cho reranker.
QUERY = "Điều kiện cấp giấy chứng nhận quyền sử dụng đất là gì?"

# Query expansion dùng cho BM25 để tăng recall.
# Reranker vẫn dùng QUERY gốc để tránh lệch intent.
USE_QUERY_EXPANSION = True

def safe_str(value: Any) -> str:
    """Convert any value to a clean string. Treat None/NaN as empty."""
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except Exception:
        pass
    return str(value)


def normalize_text(text: Any) -> str:
    """Normalize Vietnamese text for rule-based matching."""
    text = safe_str(text).lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_like_certificate_condition_query(query: str) -> bool:
    q = normalize_text(query)
    has_certificate = any(term in q for term in ["giay chung nhan", "so do", "gcn"])
    has_land = "dat" in q or "quyen su dung dat" in q
    has_intent = any(term in q for term in ["dieu kien", "duoc cap", "du dieu kien", "cap", "lam"])
    return has_certificate and has_land and has_intent


def expand_query_for_bm25(query: str) -> str:
    """Expand common legal queries to improve BM25 recall.

    Only used for BM25 retrieval. Cross-Encoder reranker still receives the original query.
    """
    if not USE_QUERY_EXPANSION:
        return query

    q = normalize_text(query)

    if looks_like_certificate_condition_query(query):
        expansion = " ".join([
            query,
            "cấp Giấy chứng nhận quyền sử dụng đất",
            "điều kiện cấp Giấy chứng nhận",
            "đủ điều kiện được cấp Giấy chứng nhận",
            "hộ gia đình cá nhân đang sử dụng đất",
            "có giấy tờ về quyền sử dụng đất",
            "không có giấy tờ về quyền sử dụng đất",
            "sử dụng đất ổn định",
            "không có tranh chấp",
            "phù hợp quy hoạch",
            "Luật Đất đai",
            "Điều 100 Điều 101 Điều 137 Điều 138",
            "sổ đỏ",
        ])
        return expansion

    if any(term in q for term in ["boi thuong", "thu hoi dat", "tai dinh cu"]):
        return " ".join([query, "bồi thường khi Nhà nước thu hồi đất hỗ trợ tái định cư điều kiện được bồi thường"])

    return query


def preview_results(results: Sequence[Dict[str, Any]], max_content_chars: int = 450) -> None:
    """Pretty print retrieval/reranking results."""
    for item in results:
        rank = item.get("rerank_rank", item.get("rank"))
        print("Rank:", rank)
        print("Chunk ID:", item.get("chunk_id"))
        print("Title:", item.get("title"))

        for key in [
            "bm25_score", "bm25_norm_score", "rerank_raw_score", "rerank_norm_score",
            "base_score", "locality_penalty", "off_intent_penalty",
            "certificate_intent_penalty", "legal_type_priority", "direct_answer_boost",
            "locality_boost", "final_score", "relative_final_score",
        ]:
            if key in item:
                print(f"{key}:", item.get(key))

        print("Content:", safe_str(item.get("content"))[:max_content_chars])
        print("-" * 100)

def load_chunks(data_path: Path) -> List[Dict[str, Any]]:
    """Load corpus chunks from parquet/csv/jsonl and return list[dict]."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {data_path.resolve()}\\n"
            "Hãy sửa DATA_PATH cho đúng vị trí file corpus của bạn."
        )

    suffix = data_path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif suffix == ".csv":
        df = pd.read_csv(data_path)
    elif suffix in {".jsonl", ".json"}:
        df = pd.read_json(data_path, lines=(suffix == ".jsonl"))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if "content" not in df.columns:
        raise ValueError("Data phải có cột 'content' để BM25 và reranker hoạt động.")

    if "chunk_id" not in df.columns:
        df = df.copy()
        df["chunk_id"] = [f"chunk_{i}" for i in range(len(df))]

    for col in [
        "document_id", "document_number", "title", "legal_type",
        "article", "clause", "point", "content", "source_url"
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    return df.to_dict(orient="records")


start = time.time()
chunks = load_chunks(DATA_PATH)
print("Load time:", round(time.time() - start, 2), "seconds")

print("\nSample chunk:")
sample = chunks[0]
for key in ["chunk_id", "document_number", "title", "legal_type", "article", "clause", "content"]:
    print(f"{key}:", safe_str(sample.get(key))[:300])