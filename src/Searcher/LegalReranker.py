from __future__ import annotations

from typing import Any, Dict, List
import math
import re
import unicodedata


LEGAL_TYPE_PRIOR = {
    "Luật": 0.08,
    "Bộ luật": 0.08,
    "Nghị định": 0.05,
    "Thông tư": 0.035,
    "Nghị quyết": 0.025,
    "Quyết định": 0.00,
    "Công văn": -0.08,
}


def safe_str(value: Any) -> str:
    """Convert any value to a clean string. Treat None/NaN as empty."""
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return str(value)


def normalize_text(text: Any) -> str:
    """Normalize Vietnamese text for rule-based legal reranking."""
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
    """Expand common legal queries when a caller explicitly wants heuristic recall."""
    q = normalize_text(query)

    if looks_like_certificate_condition_query(query):
        return " ".join([
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

    if any(term in q for term in ["boi thuong", "thu hoi dat", "tai dinh cu"]):
        return " ".join([query, "bồi thường khi Nhà nước thu hồi đất hỗ trợ tái định cư điều kiện được bồi thường"])

    return query


def intent_penalty(query: str, item: Dict[str, Any]) -> float:
    q = query.lower()
    text = f"{item.get('title', '')} {item.get('content', '')}".lower()
    legal_type = str(item.get("legal_type", "")).lower()

    penalty = 0.0
    is_certificate_condition_query = (
        "điều kiện" in q
        and "giấy chứng nhận" in q
        and "quyền sử dụng đất" in q
    )

    if is_certificate_condition_query:
        bad_terms = [
            "thu tiền sử dụng đất",
            "chính sách thu tiền sử dụng đất",
            "tổng cục thuế",
        ]
        if any(term in text for term in bad_terms):
            penalty += 0.20

        if "công văn" in legal_type or "công văn" in text[:200]:
            penalty += 0.10

    return penalty


def legal_type_prior(item: Dict[str, Any]) -> float:
    legal_type_text = str(item.get("legal_type", ""))
    title_text = str(item.get("title", ""))
    combined = f"{legal_type_text} {title_text}"

    for legal_type, prior in LEGAL_TYPE_PRIOR.items():
        if legal_type.lower() in combined.lower():
            return prior

    return 0.0


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

NORMALIZED_PROVINCES = [normalize_text(province) for province in VIETNAM_PROVINCES]


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


def apply_legal_type_prior(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for item in results:
        prior = legal_type_prior(item)
        score_before_prior = float(item.get("final_score", item.get("score", 0.0)))
        multiplier = max(0.0, 1.0 + prior)

        item["score_before_legal_type_prior"] = score_before_prior
        item["legal_type_prior"] = prior
        item["legal_type_multiplier"] = multiplier
        item["legal_type_rerank_delta"] = score_before_prior * (multiplier - 1.0)
        item["final_score"] = score_before_prior * multiplier
        item["score"] = item["final_score"]

    results.sort(key=lambda item: item.get("final_score", item.get("score", 0.0)), reverse=True)
    return results


def apply_intent_penalties(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for item in results:
        penalty = min(max(intent_penalty(query, item), 0.0), 1.0)
        score_before_intent = float(item.get("final_score", item.get("score", 0.0)))
        intent_multiplier = 1.0 - penalty

        item["score_before_intent"] = score_before_intent
        item["intent_penalty"] = penalty
        item["intent_multiplier"] = intent_multiplier
        item["intent_rerank_delta"] = score_before_intent * (intent_multiplier - 1.0)
        item["final_score"] = score_before_intent * intent_multiplier
        item["score"] = item["final_score"]

    results.sort(key=lambda item: item.get("final_score", item.get("score", 0.0)), reverse=True)
    return results
