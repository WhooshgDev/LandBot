import faiss
import pickle
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
MODEL_NAME = "BAAI/bge-m3"
MAX_SEQ_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load config
with open(f"{VECTOR_STORE_DIR}/config.json") as f:
    config = json.load(f)
print(f"Config: {config}")

# Load index
index = faiss.read_index(f"{VECTOR_STORE_DIR}/faiss_index.bin")
print(f"Index size: {index.ntotal} vectors, dimension: {index.d}")

# Load metadata
with open(f"{VECTOR_STORE_DIR}/metadata.pkl", "rb") as f:
    data = pickle.load(f)
chunk_ids = data["chunk_ids"]
metadata = data["metadata"]
print(f"Metadata: {len(chunk_ids)} entries")

# Basic sanity checks
assert index.ntotal == config["num_vectors"], "Vector count mismatch!"
assert index.d == config["dim"], "Dimension mismatch!"
assert len(chunk_ids) == len(metadata), "Metadata length mismatch!"
print("All basic checks passed!\n")

# Test query
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="sdpa", dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
model.to(DEVICE)
model.eval()

def embed_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0]
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)

test_queries = [
    "Đất không sở hữu có được cấp sổ đỏ không?",
    "Tranh chấp đất đai giải quyết như thế nào?",
    "Điều kiện để được cấp sổ đỏ là gì?",          
]

for query in test_queries:
    emb = embed_text(query)
    scores, indices = index.search(emb, k=5)

    print(f"Query: {query}")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        meta = metadata[idx]
        title = meta.get("title", "N/A")
        doc_num = meta.get("document_number", "N/A")
        art = meta.get("article", "")
        clause = meta.get("clause", "")
        point = meta.get("point", "")
        location = f"Điều {art}" + (f", Khoản {clause}" if clause else "") + (f", Điểm {point}" if point else "")
        content_preview = meta.get("content", "")[:150] if "content" in meta else ""
        print(f"  {rank+1}. [{doc_num}] {title}")
        print(f"     {location} — score: {score:.4f}")
        if content_preview:
            print(f"     Preview: {content_preview}...")
    print()

print("Vector store is working correctly!")
