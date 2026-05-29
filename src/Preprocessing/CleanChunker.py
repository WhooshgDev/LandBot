from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl


# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "LandLawDocument.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "LandLawDocumentCleaned.parquet"


# =========================================================
# Boilerplate Patterns
# =========================================================
BOILERPLATE_PATTERNS = [
    r"CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    r"Độc lập\s*-\s*Tự do\s*-\s*Hạnh phúc",
    r"Trang\s+\d+",
    r"KT\.\s*BỘ TRƯỞNG",
    r"PHÓ THỦ TƯỚNG",
    r"TM\.\s*CHÍNH PHỦ",
    r"Nơi nhận:",
]


# =========================================================
# Regex Patterns
# =========================================================
ARTICLE_PATTERN = re.compile(
    r"(?:^|\n)\s*Điều\s+(\d+[A-Za-z]*)[\.:\-]?\s*(.*?)"
    r"(?=(?:\n\s*Điều\s+\d+[A-Za-z]*)|\Z)",
    re.DOTALL | re.IGNORECASE,
)

CLAUSE_PATTERN = re.compile(
    r"(?:^|\n)\s*(\d+)\.\s*(.*?)"
    r"(?=(?:\n\s*\d+\.)|\Z)",
    re.DOTALL,
)

POINT_PATTERN = re.compile(
    r"(?:^|\n)\s*([a-z])\)\s*(.*?)"
    r"(?=(?:\n\s*[a-z]\))|\Z)",
    re.DOTALL,
)

REFERENCE_PATTERN = re.compile(
    r"Điều\s+(\d+)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)\s+(\d{4})",
    re.UNICODE,
)


ADMIN_TAIL_PATTERNS = [
    r"Chánh Văn phòng",
    r"Thủ trưởng các sở",
    r"Chủ tịch UBND các",
    r"chịu trách nhiệm thi hành",
    r"Nơi nhận",
    r"Lưu:\s*VT",
    r"TM\.\s*(ỦY BAN NHÂN DÂN|UBND|CHÍNH PHỦ)",
    r"KT\.\s*CHỦ TỊCH",
    r"PHÓ CHỦ TỊCH",
    r"CHỦ TỊCH\s+[A-ZÀ-Ỹ]",
]

TABLE_SIGNAL_PATTERNS = [
    r"\|.*\|",
    r"\bSTT\b",
    r"\bĐơn vị tính\b",
    r"\bTổng diện tích\b",
    r"\bGiá đất\b",
    r"\bGIÁ ĐẤT\b",
    r"\bđồng/m\s*2\b",
    r"\bHa\b",
]


def is_table_like(content: str) -> bool:
    if not content:
        return False

    table_line_count = sum(
        1 for line in content.splitlines()
        if line.strip().startswith("|") and line.strip().endswith("|")
    )

    if table_line_count >= 3:
        return True

    return any(
        re.search(pattern, content, flags=re.IGNORECASE)
        for pattern in TABLE_SIGNAL_PATTERNS
    )


def is_admin_tail_like(content: str) -> bool:
    if not content:
        return False

    # Do not remove tables even if they contain some admin-looking words.
    if is_table_like(content):
        return False

    matches = 0

    for pattern in ADMIN_TAIL_PATTERNS:
        if re.search(pattern, content, flags=re.IGNORECASE):
            matches += 1

    # Strong admin-tail signal
    if matches >= 2:
        return True

    # Very common pure implementation clause
    if re.search(
        r"chịu trách nhiệm thi hành.*(Quyết định|Nghị quyết|Thông tư) này",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return True

    return False


# =========================================================
# Text Cleaning
# =========================================================
def normalize_unicode(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    replacements = {
        "\u200b": " ",
        "\xa0": " ",
        "\ufeff": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text



def remove_boilerplate(text: str) -> str:
    if not text:
        return ""

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    return text



def normalize_whitespace(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()



def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_boilerplate(text)
    text = normalize_whitespace(text)

    return text


# =========================================================
# Metadata Helpers
# =========================================================
def safe_get(row: Dict, key: str) -> Optional[str]:
    value = row.get(key)

    if value is None:
        return None

    return str(value)



def generate_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# =========================================================
# Reference Extraction
# =========================================================
def extract_references(text: str) -> List[Dict]:
    references = []

    for match in REFERENCE_PATTERN.finditer(text):
        article, law_name, year = match.groups()

        references.append(
            {
                "article": article,
                "law_name": law_name.strip(),
                "year": year,
            }
        )

    return references


# =========================================================
# Legal Structure Parsing
# =========================================================
def parse_points(
    point_text: str,
    base_metadata: Dict,
    article_id: str,
    clause_id: str,
) -> List[Dict]:
    chunks = []

    point_matches = list(POINT_PATTERN.finditer(point_text))

    if not point_matches:
        chunk_text = normalize_whitespace(point_text)

        if chunk_text:
            chunks.append(
                {
                    **base_metadata,
                    "article": article_id,
                    "clause": clause_id,
                    "point": None,
                    "content": chunk_text,
                    "references": extract_references(chunk_text),
                }
            )

        return chunks

    parsed_points = []

    for point_match in point_matches:
        point_id, point_content = point_match.groups()
        point_content = normalize_whitespace(point_content)

        if not point_content:
            continue

        parsed_points.append(
            {
                "point": point_id,
                "content": point_content,
            }
        )

    if not parsed_points:
        return chunks

    # If any Điểm is too short, group all Điểm under this Khoản.
    # This avoids creating tiny fragments like:
    # a) Diện tích thửa đất ≥ 36 m2;
    # b) Chiều rộng mặt tiền ≥ 4 m;
    # These are better retrieved together as one Khoản-level chunk.
    has_tiny_point = any(
        len(item["content"]) < 50
        for item in parsed_points
    )

    if has_tiny_point:
        grouped_content = normalize_whitespace(point_text)

        if grouped_content:
            chunks.append(
                {
                    **base_metadata,
                    "article": article_id,
                    "clause": clause_id,
                    "point": None,
                    "content": grouped_content,
                    "references": extract_references(grouped_content),
                }
            )

        return chunks

    for item in parsed_points:
        chunks.append(
            {
                **base_metadata,
                "article": article_id,
                "clause": clause_id,
                "point": item["point"],
                "content": item["content"],
                "references": extract_references(item["content"]),
            }
        )

    return chunks



def parse_clauses(
    article_text: str,
    base_metadata: Dict,
    article_id: str,
) -> List[Dict]:
    chunks = []

    clause_matches = list(CLAUSE_PATTERN.finditer(article_text))

    if not clause_matches:
        content = normalize_whitespace(article_text)

        if content:
            chunks.append(
                {
                    **base_metadata,
                    "article": article_id,
                    "clause": None,
                    "point": None,
                    "content": content,
                    "references": extract_references(content),
                }
            )

        return chunks

    for clause_match in clause_matches:
        clause_id, clause_content = clause_match.groups()

        clause_chunks = parse_points(
            point_text=clause_content,
            base_metadata=base_metadata,
            article_id=article_id,
            clause_id=clause_id,
        )

        chunks.extend(clause_chunks)

    return chunks



def parse_articles(text: str, metadata: Dict) -> List[Dict]:
    chunks = []

    article_matches = list(ARTICLE_PATTERN.finditer(text))

    if not article_matches:
        cleaned_text = normalize_whitespace(text)

        if cleaned_text:
            chunks.append(
                {
                    **metadata,
                    "article": None,
                    "clause": None,
                    "point": None,
                    "content": cleaned_text,
                    "references": extract_references(cleaned_text),
                }
            )

        return chunks

    for article_match in article_matches:
        article_id, article_content = article_match.groups()

        article_chunks = parse_clauses(
            article_text=article_content,
            base_metadata=metadata,
            article_id=article_id,
        )

        chunks.extend(article_chunks)

    return chunks


# =========================================================
# Main Processing
# =========================================================
def process_document(row: Dict) -> List[Dict]:
    content = (
        safe_get(row, "content")
        or safe_get(row, "text")
        or safe_get(row, "full_text")
        or ""
    )

    cleaned_text = clean_text(content)

    if not cleaned_text:
        return []

    metadata = {
        "document_id": safe_get(row, "id"),
        "document_number": safe_get(row, "document_number"),
        "title": safe_get(row, "title"),
        "legal_type": safe_get(row, "legal_type"),
        "issue_date": safe_get(row, "issue_date"),
        "effective_date": safe_get(row, "effective_date"),
        "source_url": safe_get(row, "url"),
    }

    chunks = parse_articles(cleaned_text, metadata)

    cleaned_chunks = []

    for chunk in chunks:
        content = chunk.get("content", "")

        if is_admin_tail_like(content):
            continue

        if is_table_like(content):
            chunk["chunk_type"] = "table"
        else:
            chunk["chunk_type"] = "text"

        cleaned_chunks.append(chunk)

    chunks = cleaned_chunks

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{metadata['document_id']}_{idx}"

        chunk["chunk_id"] = chunk_id
        chunk["content_hash"] = generate_hash(chunk["content"])

    return chunks


# =========================================================
# Deduplication
# =========================================================
def deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique(subset=["content_hash"])


# =========================================================
# Main
# =========================================================
def main() -> None:
    print("Loading parquet...")

    df = pl.read_parquet(INPUT_PATH)

    print(f"Loaded {len(df):,} documents")

    all_chunks = []

    rows = df.to_dicts()

    for idx, row in enumerate(rows):
        try:
            chunks = process_document(row)
            all_chunks.extend(chunks)

            if idx % 1000 == 0:
                print(f"Processed {idx:,}/{len(rows):,} documents")

        except Exception as e:
            print(f"Error processing row {idx}: {e}")

    if not all_chunks:
        raise ValueError("No chunks were generated.")

    print(f"Generated {len(all_chunks):,} chunks before deduplication")

    cleaned_df = pl.DataFrame(all_chunks)

    cleaned_df = deduplicate(cleaned_df)

    print(f"Remaining {len(cleaned_df):,} chunks after deduplication")


    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.write_parquet(OUTPUT_PATH)

    print(f"Saved cleaned parquet to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
