import os
import re
import json
import time
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# Config
# =========================================================

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "LandLawDocumentCleaned.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval_queries.jsonl"

MODEL_NAME = os.getenv("GEN_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

NUM_SOURCE_CHUNKS = 300
QUERIES_PER_CHUNK = 3

MIN_CONTENT_LEN = 120
MAX_CONTENT_LEN = 1200

RANDOM_SEED = 42
SLEEP_SECONDS = 0.2

random.seed(RANDOM_SEED)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

VIETNAM_PROVINCES = [
    "Hà Nội", "TP.HCM", "Thành phố Hồ Chí Minh", "Hồ Chí Minh",
    "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Đồng Tháp", "Ninh Thuận",
    "Quảng Nam", "Bình Dương", "Thừa Thiên Huế", "Lâm Đồng",
    "Đồng Nai", "Long An", "Tiền Giang", "Bến Tre", "Trà Vinh",
    "Vĩnh Long", "An Giang", "Kiên Giang", "Hậu Giang", "Sóc Trăng",
    "Bạc Liêu", "Cà Mau", "Tây Ninh", "Bình Phước", "Bà Rịa",
    "Vũng Tàu", "Bình Thuận", "Khánh Hòa", "Phú Yên", "Bình Định",
    "Quảng Ngãi", "Quảng Trị", "Quảng Bình", "Hà Tĩnh", "Nghệ An",
    "Thanh Hóa", "Nam Định", "Ninh Bình", "Thái Bình", "Hưng Yên",
    "Hải Dương", "Bắc Ninh", "Bắc Giang", "Vĩnh Phúc", "Phú Thọ",
    "Thái Nguyên", "Lạng Sơn", "Cao Bằng", "Bắc Kạn", "Hà Giang",
    "Tuyên Quang", "Yên Bái", "Lào Cai", "Sơn La", "Điện Biên",
    "Lai Châu", "Hòa Bình", "Gia Lai", "Kon Tum", "Đắk Lắk",
    "Đắk Nông",
]

STOPWORDS = {
    "của", "cho", "với", "trong", "theo", "được", "khi", "nào", "như",
    "thế", "nào", "tôi", "nếu", "về", "và", "hoặc", "các", "những",
    "một", "người", "đất", "quyền", "sử", "dụng", "phải", "cần", "làm",
    "gì", "ra", "sao", "quy", "định", "pháp", "luật",
}


# =========================================================
# Helpers
# =========================================================

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def meaningful_terms(text: str) -> set[str]:
    terms = set()
    for term in re.findall(r"[A-Za-zÀ-ỹà-ỹ0-9]+", normalize_text(text)):
        if len(term) < 4:
            continue
        if term in STOPWORDS:
            continue
        terms.add(term)
    return terms


def extract_localities(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [province for province in VIETNAM_PROVINCES if normalize_text(province) in normalized]


def is_local_document(metadata_text: str) -> bool:
    normalized = normalize_text(metadata_text)
    local_markers = [
        "ubnd", "ủy ban nhân dân", "uỷ ban nhân dân", "hđnd",
        "hội đồng nhân dân", "trên địa bàn", "tỉnh", "thành phố",
    ]
    return any(marker in normalized for marker in local_markers) or bool(extract_localities(metadata_text))


def has_required_locality(query: str, metadata_text: str) -> bool:
    localities = extract_localities(metadata_text)
    if not localities:
        return True
    normalized_query = normalize_text(query)
    return any(normalize_text(locality) in normalized_query for locality in localities)


def has_enough_anchors(query: str, source_text: str, metadata_text: str) -> bool:
    query_terms = meaningful_terms(query)
    source_terms = meaningful_terms(f"{metadata_text} {source_text}")
    if not query_terms:
        return False

    overlap = query_terms & source_terms
    if len(overlap) >= 3:
        return True

    normalized_query = normalize_text(query)
    distinctive_phrases = [
        "giấy chứng nhận",
        "bồi thường",
        "hỗ trợ",
        "tái định cư",
        "thu hồi đất",
        "tặng cho",
        "đăng ký biến động",
        "chuyển mục đích",
        "giao đất",
        "cho thuê đất",
    ]
    return bool(overlap) and any(phrase in normalized_query for phrase in distinctive_phrases)


def looks_like_bad_chunk(text: str) -> bool:
    """
    Remove signatures, recipient lists, appendix noise, and administrative endings.
    Adjust this as you inspect your dataset.
    """
    t = text.lower()

    bad_patterns = [
        "nơi nhận",
        "lưu:",
        "kt. chủ tịch",
        "tm. ủy ban",
        "chánh văn phòng",
        "phó chủ tịch",
        "đã ký",
        "ký tên",
        "danh sách",
        "phụ lục",
        "mẫu số",
        "cộng hòa xã hội chủ nghĩa việt nam",
        "độc lập - tự do - hạnh phúc",
    ]

    if any(p in t for p in bad_patterns):
        return True

    # Too many line breaks usually means table/list/signature noise
    if text.count("\n") > 20:
        return True

    # Too little Vietnamese alphabetic content
    letters = re.findall(r"[A-Za-zÀ-ỹà-ỹ]", text)
    if len(letters) < 50:
        return True

    return False


def sample_good_chunks(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Sample chunks likely to produce useful legal queries.
    """
    df = df.copy()

    if "content_length" not in df.columns:
        df["content_length"] = df["content"].astype(str).str.len()

    df = df[
        (df["content_length"] >= MIN_CONTENT_LEN)
        & (df["content_length"] <= MAX_CONTENT_LEN)
    ]

    # Prefer chunks with article info
    if "article" in df.columns:
        df = df[df["article"].notna()]

    df["content"] = df["content"].astype(str)
    df = df[~df["content"].apply(looks_like_bad_chunk)]

    if len(df) < n:
        print(f"Warning: only {len(df)} good chunks found. Sampling all.")
        return df.sample(frac=1, random_state=RANDOM_SEED)

    return df.sample(n=n, random_state=RANDOM_SEED)


def extract_json(text: str):
    """
    Robust JSON extraction in case the model wraps output in text.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")

    return json.loads(text[start:end + 1])


def is_query_too_extractive(query: str, source_text: str) -> bool:
    """
    Simple overlap filter.
    Rejects queries that copy too much from the source chunk.
    """
    q_words = set(re.findall(r"\w+", query.lower()))
    s_words = set(re.findall(r"\w+", source_text.lower()))

    if not q_words:
        return True

    overlap = len(q_words & s_words) / len(q_words)

    # Keyword queries naturally overlap more, so this threshold is not too strict.
    return overlap > 0.85


def valid_query(q: str, source_text: str, metadata_text: str = "") -> bool:
    q = q.strip()

    if len(q) < 10:
        return False

    if len(q) > 250:
        return False

    banned_phrases = [
        "đoạn văn",
        "văn bản trên",
        "nội dung trên",
        "the passage",
        "provided text",
        "according to the text",
        "the given passage",
    ]

    if any(p in q.lower() for p in banned_phrases):
        return False

    if is_query_too_extractive(q, source_text):
        return False

    if is_local_document(metadata_text) and not has_required_locality(q, metadata_text):
        return False

    if not has_enough_anchors(q, source_text, metadata_text):
        return False

    return True


# =========================================================
# Prompt
# =========================================================

def build_generation_prompt(row) -> str:
    chunk_id = safe_str(row.get("chunk_id", ""))
    document_id = safe_str(row.get("document_id", ""))
    document_number = safe_str(row.get("document_number", ""))
    title = safe_str(row.get("title", ""))
    article = safe_str(row.get("article", ""))
    clause = safe_str(row.get("clause", ""))
    point = safe_str(row.get("point", ""))
    content = safe_str(row.get("content", ""))

    return f"""
You are creating a synthetic evaluation dataset for Vietnamese Land Law document retrieval.

Given one legal passage, generate {QUERIES_PER_CHUNK} realistic Vietnamese search queries that this passage can answer.

Important requirements:
- Queries must be in Vietnamese.
- Queries must be answerable using the legal passage.
- Queries must be specific enough that this passage is the best retrieval target, not just one of many possible legal answers.
- If the metadata title/document is local to a province or city, include that locality in every query.
- Include at least one distinctive legal anchor from the passage, such as the procedure, document name, article/clause, agency, money/support type, registration action, or land-use action.
- Do NOT copy long exact phrases from the passage.
- Do NOT mention "the passage", "the text", or "the provided document".
- Do NOT include answers.
- Make the queries realistic: citizen question, legal student question, or search-engine style query.
- Generate diverse query types:
  1. direct: a direct legal question
  2. scenario: a practical real-life situation
  3. keyword: short keyword-style search query

Return JSON only in this exact structure:
{{
  "queries": [
    {{
      "query": "...",
      "query_type": "direct",
      "difficulty": "easy"
    }},
    {{
      "query": "...",
      "query_type": "scenario",
      "difficulty": "medium"
    }},
    {{
      "query": "...",
      "query_type": "keyword",
      "difficulty": "easy"
    }}
  ]
}}

Metadata:
chunk_id: {chunk_id}
document_id: {document_id}
document_number: {document_number}
title: {title}
article: {article}
clause: {clause}
point: {point}

Legal passage:
\"\"\"
{content}
\"\"\"
""".strip()


def call_llm(prompt: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate high-quality Vietnamese legal retrieval queries and always return valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )

            text = response.choices[0].message.content
            return extract_json(text)

        except Exception as e:
            print(f"LLM call failed, attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(2 ** attempt)

    return None


# =========================================================
# Main generation
# =========================================================

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    print(f"Rows loaded: {len(df):,}")
    sampled = sample_good_chunks(df, NUM_SOURCE_CHUNKS)
    print(f"Sampled chunks: {len(sampled):,}")

    query_id_counter = 1
    total_written = 0
    total_failed = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for _, row in tqdm(sampled.iterrows(), total=len(sampled)):
            prompt = build_generation_prompt(row)
            result = call_llm(prompt)

            if result is None or "queries" not in result:
                total_failed += 1
                continue

            source_text = safe_str(row.get("content", ""))

            chunk_id = safe_str(row.get("chunk_id", ""))
            document_id = safe_str(row.get("document_id", ""))
            document_number = safe_str(row.get("document_number", ""))
            title = safe_str(row.get("title", ""))
            article = safe_str(row.get("article", ""))
            clause = safe_str(row.get("clause", ""))
            point = safe_str(row.get("point", ""))
            metadata_text = " ".join([document_id, document_number, title, article, clause, point])

            for item in result.get("queries", []):
                query = safe_str(item.get("query", ""))
                query_type = safe_str(item.get("query_type", "unknown"))
                difficulty = safe_str(item.get("difficulty", "unknown"))

                if not valid_query(query, source_text, metadata_text):
                    continue

                record = {
                    "query_id": f"q{query_id_counter:06d}",
                    "query": query,
                    "query_type": query_type,
                    "difficulty": difficulty,
                    "gold_chunk_ids": [chunk_id],
                    "gold_document_ids": [document_id],
                    "gold_title": title,
                    "gold_article": article,
                    "gold_clause": clause,
                    "gold_point": point,
                    "source": "llm_synthetic_from_chunk",
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                query_id_counter += 1
                total_written += 1

            time.sleep(SLEEP_SECONDS)

    print("Done.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Queries written: {total_written:,}")
    print(f"Failed chunks: {total_failed:,}")


if __name__ == "__main__":
    main()
