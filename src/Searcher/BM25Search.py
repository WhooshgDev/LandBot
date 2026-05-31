from __future__ import annotations

from copy import deepcopy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import gc
import json
import os
import pickle
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except Exception:
    ViTokenizer = None
    PYVI_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    TQDM_AVAILABLE = False

sys.modules.setdefault("BM25Search", sys.modules[__name__])

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
BM25_INDEX_CACHE_PATH = CACHE_DIR / f"bm25_index_{TOKENIZER_NAME}.pkl"

# Nên lấy candidate rộng hơn top-10 để reranker có cơ hội tìm đúng hơn.
BM25_TOP_N = 100
FINAL_TOP_K = 10

# Query expansion dùng cho BM25 để tăng recall.
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

    return df.to_dict(orient="records")


class BM25InvertedIndex:
    """BM25 index backed by posting lists instead of full-corpus term scans."""

    def __init__(
        self,
        postings: Dict[str, Tuple[np.ndarray, np.ndarray]],
        idf: Dict[str, float],
        doc_len: np.ndarray,
        avgdl: float,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.postings = postings
        self.idf = idf
        self.doc_len = doc_len
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = int(len(doc_len))

    @classmethod
    def from_tokenized_documents(
        cls,
        tokenized_documents: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> "BM25InvertedIndex":
        corpus_size = len(tokenized_documents)
        doc_len = np.fromiter((len(doc) for doc in tokenized_documents), dtype=np.float32, count=corpus_size)
        avgdl = float(doc_len.mean()) if corpus_size else 0.0

        posting_builders: Dict[str, List[List[int]]] = defaultdict(lambda: [[], []])
        document_frequency: Dict[str, int] = defaultdict(int)

        for doc_id, document in enumerate(tokenized_documents):
            frequencies = Counter(document)
            for term, frequency in frequencies.items():
                posting_builders[term][0].append(doc_id)
                posting_builders[term][1].append(frequency)
                document_frequency[term] += 1

        idf: Dict[str, float] = {}
        negative_idf_terms = []
        idf_sum = 0.0
        for term, frequency in document_frequency.items():
            term_idf = np.log(corpus_size - frequency + 0.5) - np.log(frequency + 0.5)
            term_idf = float(term_idf)
            idf[term] = term_idf
            idf_sum += term_idf
            if term_idf < 0:
                negative_idf_terms.append(term)

        average_idf = idf_sum / len(idf) if idf else 0.0
        epsilon_idf = epsilon * average_idf
        for term in negative_idf_terms:
            idf[term] = epsilon_idf

        postings = {
            term: (
                np.asarray(values[0], dtype=np.int32),
                np.asarray(values[1], dtype=np.float32),
            )
            for term, values in posting_builders.items()
        }

        return cls(
            postings=postings,
            idf=idf,
            doc_len=doc_len,
            avgdl=avgdl,
            k1=k1,
            b=b,
            epsilon=epsilon,
        )

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if self.corpus_size == 0:
            return scores

        length_norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for term in query:
            posting = self.postings.get(term)
            term_idf = self.idf.get(term, 0.0)
            if posting is None or term_idf == 0.0:
                continue

            doc_ids, frequencies = posting
            scores[doc_ids] += term_idf * (
                frequencies * (self.k1 + 1) / (frequencies + length_norm[doc_ids])
            )

        return scores

class BM25Retriever:
    """
    BM25 lexical retriever for legal chunks.

    BM25 is used as first-stage retrieval:
        query -> top-N candidate chunks

    Then CrossEncoderLegalReranker reranks only those top-N candidates.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        tokenizer: str = "pyvi",
        use_cache: bool = True,
        cache_path: Optional[Path] = None,
        index_cache_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
    ):
        self.chunks = chunks
        self.num_documents = len(chunks)
        self.tokenizer = tokenizer
        self.use_cache = use_cache
        self.cache_path = cache_path
        self.index_cache_path = index_cache_path
        self.data_path = data_path

        self.documents: List[str] = []
        self.bm25 = self._load_bm25_index_from_cache()
        if self.bm25 is None:
            self.documents = [self.build_search_text(chunk) for chunk in self.chunks]
            tokenized_documents = self._load_or_tokenize_documents()
            self.bm25 = self._build_and_cache_bm25_index(tokenized_documents)
        self.documents = []

    def build_search_text(self, chunk: Dict[str, Any]) -> str:
        """Combine important metadata and content for BM25 search."""
        parts = [
            safe_str(chunk.get("document_number")),
            safe_str(chunk.get("legal_type")),
            safe_str(chunk.get("title")),
            f"Điều {safe_str(chunk.get('article'))}" if safe_str(chunk.get("article")) else "",
            f"Khoản {safe_str(chunk.get('clause'))}" if safe_str(chunk.get("clause")) else "",
            f"Điểm {safe_str(chunk.get('point'))}" if safe_str(chunk.get("point")) else "",
            safe_str(chunk.get("content")),
        ]
        return " ".join(part for part in parts if part).strip()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Vietnamese text. Falls back to simple whitespace tokenization."""
        text = safe_str(text).lower()

        if self.tokenizer == "pyvi" and PYVI_AVAILABLE:
            return ViTokenizer.tokenize(text).split()

        return text.split()

    def _load_or_tokenize_documents(self) -> List[List[str]]:
        """Load tokenized documents from cache or tokenize corpus."""
        force_rebuild = os.getenv("BM25_FORCE_REBUILD_CACHE", "").lower() in {"1", "true", "yes"}

        if force_rebuild:
            print("BM25_FORCE_REBUILD_CACHE is set. Rebuilding token cache without loading old pickle.")
        elif self.use_cache and self.cache_path is not None and self.cache_path.exists():
            metadata_status = self._cache_metadata_status()
            if metadata_status == "valid" or metadata_status == "missing":
                print(f"Loading tokenized corpus from cache: {self.cache_path}")
                start = time.time()
                with open(self.cache_path, "rb") as f:
                    cached = pickle.load(f)
                print("Cache load time:", round(time.time() - start, 2), "seconds")
                if len(cached) == len(self.documents):
                    if metadata_status == "missing":
                        self._write_cache_metadata()
                    return cached
                print(
                    "Cache length does not match current corpus. "
                    f"cached={len(cached):,}, current={len(self.documents):,}. Retokenizing..."
                )
                del cached
                gc.collect()
            else:
                print(f"Token cache metadata is stale ({metadata_status}). Retokenizing...")

        print(f"Tokenizing {len(self.documents):,} documents with tokenizer='{self.tokenizer}'...")
        start = time.time()

        documents = self.documents
        if TQDM_AVAILABLE:
            documents = tqdm(self.documents, total=len(self.documents), desc="BM25 tokenizing", unit="doc")

        tokenized_documents = [self.tokenize(doc) for doc in documents]

        print("Tokenization time:", round(time.time() - start, 2), "seconds")

        if self.use_cache and self.cache_path is not None:
            print(f"Saving tokenized corpus to cache: {self.cache_path}")
            tmp_cache_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            with open(tmp_cache_path, "wb") as f:
                pickle.dump(tokenized_documents, f)
            tmp_cache_path.replace(self.cache_path)
            self._write_cache_metadata()

        return tokenized_documents

    def _load_bm25_index_from_cache(self) -> Optional[BM25InvertedIndex]:
        force_rebuild = os.getenv("BM25_FORCE_REBUILD_CACHE", "").lower() in {"1", "true", "yes"}
        if force_rebuild:
            return None

        if (
            self.use_cache
            and self.index_cache_path is not None
            and self.index_cache_path.exists()
        ):
            metadata_status = self._cache_metadata_status(self.index_cache_path)
            if metadata_status == "valid":
                print(f"Loading BM25 inverted index from cache: {self.index_cache_path}")
                start = time.time()
                with open(self.index_cache_path, "rb") as f:
                    cached_index = pickle.load(f)
                print("BM25 index load time:", round(time.time() - start, 2), "seconds")
                if cached_index.corpus_size == self.num_documents:
                    return cached_index
                print(
                    "BM25 index length does not match current corpus. "
                    f"cached={cached_index.corpus_size:,}, current={self.num_documents:,}. Rebuilding..."
                )
                del cached_index
                gc.collect()
            else:
                print(f"BM25 index metadata is stale ({metadata_status}). Rebuilding...")

        return None

    def _build_and_cache_bm25_index(self, tokenized_documents: Sequence[Sequence[str]]) -> BM25InvertedIndex:
        print(f"Building BM25 inverted index for {self.num_documents:,} documents...")
        start = time.time()
        bm25_index = BM25InvertedIndex.from_tokenized_documents(tokenized_documents)
        print("BM25 index build time:", round(time.time() - start, 2), "seconds")

        if self.use_cache and self.index_cache_path is not None:
            print(f"Saving BM25 inverted index to cache: {self.index_cache_path}")
            tmp_index_path = self.index_cache_path.with_suffix(self.index_cache_path.suffix + ".tmp")
            with open(tmp_index_path, "wb") as f:
                pickle.dump(bm25_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_index_path.replace(self.index_cache_path)
            self._write_cache_metadata(self.index_cache_path)

        return bm25_index

    def _cache_metadata_path(self, cache_path: Optional[Path] = None) -> Optional[Path]:
        cache_path = self.cache_path if cache_path is None else cache_path
        if cache_path is None:
            return None
        return cache_path.with_suffix(cache_path.suffix + ".meta.json")

    def _current_cache_metadata(self) -> Dict[str, Any]:
        data_stat = self.data_path.stat() if self.data_path is not None and self.data_path.exists() else None
        return {
            "tokenizer": self.tokenizer,
            "num_documents": self.num_documents,
            "data_path": str(self.data_path.resolve()) if self.data_path is not None else "",
            "data_size": data_stat.st_size if data_stat is not None else None,
            "data_mtime_ns": data_stat.st_mtime_ns if data_stat is not None else None,
        }

    def _cache_metadata_status(self, cache_path: Optional[Path] = None) -> str:
        metadata_path = self._cache_metadata_path(cache_path)
        if metadata_path is None or not metadata_path.exists():
            return "missing"

        try:
            with open(metadata_path, encoding="utf-8") as f:
                cached_metadata = json.load(f)
        except Exception as exc:
            return f"unreadable metadata: {exc}"

        current_metadata = self._current_cache_metadata()
        for key, current_value in current_metadata.items():
            if cached_metadata.get(key) != current_value:
                return f"{key} mismatch"
        return "valid"

    def _write_cache_metadata(self, cache_path: Optional[Path] = None) -> None:
        metadata_path = self._cache_metadata_path(cache_path)
        if metadata_path is None:
            return

        tmp_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        with open(tmp_metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._current_cache_metadata(), f, ensure_ascii=False, indent=2)
        tmp_metadata_path.replace(metadata_path)

    def search(self, query: str, top_n: int = 100, expand_query: bool = False) -> List[Dict[str, Any]]:
        """Return top-N BM25 candidates."""
        if not self.chunks or top_n <= 0:
            return []

        if expand_query:
            query = expand_query_for_bm25(query)

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = np.asarray(self.bm25.get_scores(tokenized_query), dtype=float)

        top_n = min(top_n, len(scores))
        top_indices = np.argpartition(scores, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            idx = int(idx)
            item = deepcopy(self.chunks[idx])
            item["rank"] = rank
            item["bm25_score"] = float(scores[idx])
            item["score"] = float(scores[idx])
            item["source"] = "bm25"
            results.append(item)

        return results

    def __call__(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        return self.search(query, top_n=top_k, expand_query=True)

VIETNAM_PROVINCES = [
    "Hà Nội", "TP.HCM", "Thành phố Hồ Chí Minh", "Hồ Chí Minh",
    "Đà Nẵng", "Hải Phòng", "Cần Thơ",
    "Đồng Tháp", "Ninh Thuận", "Quảng Nam", "Bình Dương",
    "Thừa Thiên Huế", "Lâm Đồng", "Đồng Nai", "Long An",
    "Tiền Giang", "Bến Tre", "Trà Vinh", "Vĩnh Long",
    "An Giang", "Kiên Giang", "Hậu Giang", "Sóc Trăng",
    "Bạc Liêu", "Cà Mau", "Tây Ninh", "Bình Phước",
    "Bà Rịa", "Vũng Tàu", "Bình Thuận", "Khánh Hòa",
    "Phú Yên", "Bình Định", "Quảng Ngãi", "Quảng Trị",
    "Quảng Bình", "Hà Tĩnh", "Nghệ An", "Thanh Hóa",
    "Nam Định", "Ninh Bình", "Thái Bình", "Hưng Yên",
    "Hải Dương", "Bắc Ninh", "Bắc Giang", "Vĩnh Phúc",
    "Phú Thọ", "Thái Nguyên", "Lạng Sơn", "Cao Bằng",
    "Bắc Kạn", "Hà Giang", "Tuyên Quang", "Yên Bái",
    "Lào Cai", "Sơn La", "Điện Biên", "Lai Châu",
    "Hòa Bình", "Gia Lai", "Kon Tum", "Đắk Lắk", "Đắk Nông",
]

NORMALIZED_PROVINCES = [normalize_text(p) for p in VIETNAM_PROVINCES]


def query_mentions_locality(query: str) -> bool:
    q = normalize_text(query)
    if any(province in q for province in NORMALIZED_PROVINCES):
        return True
    locality_markers = [
        "tren dia ban", "dia ban", "ubnd", "uy ban nhan dan",
        "hoi dong nhan dan", "hdnd", "tinh ", "thanh pho ", "tp ",
    ]
    return any(marker in q for marker in locality_markers)


def get_query_localities(query: str) -> List[str]:
    q = normalize_text(query)
    return [province for province in NORMALIZED_PROVINCES if province in q]


def is_local_legal_document(chunk: Dict[str, Any]) -> bool:
    combined = normalize_text(
        f"{chunk.get('title', '')} {chunk.get('legal_type', '')} {chunk.get('document_number', '')}"
    )
    local_markers = [
        "ubnd", "qd-ubnd", "qđ-ubnd", "uy ban nhan dan", "hoi dong nhan dan", "hdnd",
        "tren dia ban", "dia ban tinh", "dia ban thanh pho", "dia ban tp",
    ]
    if any(marker in combined for marker in local_markers):
        return True
    if any(province in combined for province in NORMALIZED_PROVINCES):
        return True
    return False


def locality_penalty_factor(query: str, chunk: Dict[str, Any]) -> float:
    if not query_mentions_locality(query) and is_local_legal_document(chunk):
        return 0.55
    return 1.0


def locality_boost_factor(query: str, chunk: Dict[str, Any]) -> float:
    query_localities = get_query_localities(query)
    if not query_localities:
        return 1.0
    combined = normalize_text(
        f"{chunk.get('title', '')} {chunk.get('legal_type', '')} {chunk.get('document_number', '')}"
    )
    if any(locality in combined for locality in query_localities):
        return 1.15
    return 1.0


def off_intent_penalty_factor(query: str, chunk: Dict[str, Any]) -> float:
    q = normalize_text(query)
    title = normalize_text(chunk.get("title"))
    content = normalize_text(chunk.get("content"))

    off_intent_terms = ["boi thuong", "ho tro", "tai dinh cu", "thu hoi dat"]
    query_mentions_off_intent = any(term in q for term in off_intent_terms)
    title_mentions_off_intent = any(term in title for term in off_intent_terms)
    content_mentions_many = sum(term in content for term in off_intent_terms) >= 2

    if not query_mentions_off_intent and title_mentions_off_intent:
        return 0.65
    if not query_mentions_off_intent and content_mentions_many:
        return 0.80
    return 1.0


def general_certificate_query(query: str) -> bool:
    return looks_like_certificate_condition_query(query)


def certificate_intent_penalty_factor(query: str, chunk: Dict[str, Any]) -> float:
    if not general_certificate_query(query):
        return 1.0

    combined = normalize_text(
        f"{chunk.get('title', '')} {chunk.get('legal_type', '')} {chunk.get('content', '')}"
    )

    strong_bad_terms = [
        "phi tham dinh", "thu tien su dung dat", "tien su dung dat",
        "du an", "vilg", "bao cao ket qua", "tang cho",
        "nha tinh nghia", "nha tinh thuong", "nha dai doan ket", "cong van",
    ]
    medium_bad_terms = [
        "ho so dia chinh", "mau giay chung nhan", "ghi tren giay chung nhan",
        "cap doi", "cap lai", "dang ky bien dong", "chinh ly",
        "thua dat da duoc cap giay chung nhan",
    ]

    if any(term in combined for term in strong_bad_terms):
        return 0.45
    if any(term in combined for term in medium_bad_terms):
        return 0.65
    return 1.0


def direct_answer_boost_factor(query: str, chunk: Dict[str, Any]) -> float:
    if not general_certificate_query(query):
        return 1.0

    combined = normalize_text(
        f"{chunk.get('title', '')} {chunk.get('content', '')}"
    )

    positive_terms = [
        "du dieu kien duoc cap giay chung nhan",
        "du dieu kien cap giay chung nhan",
        "duoc cap giay chung nhan",
        "co giay to ve quyen su dung dat",
        "khong co giay to ve quyen su dung dat",
        "su dung dat on dinh",
        "khong co tranh chap",
        "phu hop voi quy hoach",
        "nguon goc va thoi diem su dung dat",
        "dieu kien cap giay chung nhan",
    ]

    hit_count = sum(term in combined for term in positive_terms)
    if hit_count >= 3:
        return 1.25
    if hit_count == 2:
        return 1.15
    if hit_count == 1:
        return 1.08
    return 1.0


def legal_type_priority_factor(chunk: Dict[str, Any]) -> float:
    combined = normalize_text(
        f"{chunk.get('title', '')} {chunk.get('legal_type', '')} {chunk.get('document_number', '')}"
    )

    if "van ban hop nhat" in combined and "luat dat dai" in combined:
        return 1.18
    if "luat dat dai" in combined:
        return 1.20
    if "nghi dinh" in combined:
        return 1.08
    if "thong tu" in combined:
        return 1.00
    if "cong van" in combined:
        return 0.55
    if "du an" in combined or "ke hoach" in combined or "vilg" in combined:
        return 0.55
    if "quyet dinh" in combined:
        return 0.75
    return 1.0


_default_retriever: Optional[BM25Retriever] = None


def get_default_retriever() -> BM25Retriever:
    global _default_retriever
    if _default_retriever is None:
        chunks = load_chunks(DATA_PATH)
        _default_retriever = BM25Retriever(
            chunks=chunks,
            tokenizer=TOKENIZER_NAME,
            use_cache=USE_TOKEN_CACHE,
            cache_path=TOKEN_CACHE_PATH,
            index_cache_path=BM25_INDEX_CACHE_PATH,
            data_path=DATA_PATH,
        )
    return _default_retriever


def bm25_search(query: str, top_k: int = 100) -> List[Dict[str, Any]]:
    return get_default_retriever().search(query, top_n=top_k, expand_query=True)


if __name__ == "__main__":
    start = time.time()
    retriever = get_default_retriever()
    print("BM25 build/load time:", round(time.time() - start, 2), "seconds")
    print("PyVi available:", PYVI_AVAILABLE)
