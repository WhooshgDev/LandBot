import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import faiss
import os
import pickle
import gc
import json
import threading
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_NAME = "BAAI/bge-m3"
MAX_SEQ_LEN = 512
BATCH_SIZE_PER_GPU = 512
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
DATA_PATH = PROJECT_ROOT / "data" / "LandLawDocumentCleaned.parquet"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

EMBEDDING_TEXT_COLUMNS = [
    "document_number",
    "legal_type",
    "title",
    "article",
    "clause",
    "point",
    "content",
]


def safe_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def build_embedding_text(row: pd.Series) -> str:
    parts = [
        safe_text(row.get("document_number")),
        safe_text(row.get("legal_type")),
        safe_text(row.get("title")),
        f"Điều {safe_text(row.get('article'))}" if safe_text(row.get("article")) else "",
        f"Khoản {safe_text(row.get('clause'))}" if safe_text(row.get("clause")) else "",
        f"Điểm {safe_text(row.get('point'))}" if safe_text(row.get("point")) else "",
        safe_text(row.get("content")),
    ]
    return " ".join(part for part in parts if part)

print(f"GPUs: {NUM_GPUS}")
for i in range(NUM_GPUS):
    vram = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | VRAM: {vram:.1f}GB")

# Load data
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} chunks")
content = df.apply(build_embedding_text, axis=1).tolist()
ids = df["chunk_id"].tolist()
metadata = df.to_dict(orient="records")

# Load model(s)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_model(device_id):
    device = torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")
    m = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="sdpa", dtype=torch.float16)
    m.to(device)
    m.eval()
    print(f"  Model loaded on {device}")
    return m, device

models = [load_model(i) for i in range(NUM_GPUS)] if NUM_GPUS > 0 else [(AutoModel.from_pretrained(MODEL_NAME), torch.device("cpu"))]

# Embedding function
def embed_chunk(texts, model, device):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        embs = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
    return embs.cpu().numpy().astype(np.float32)

def process_on_gpu(gpu_id, texts_chunk, result_list, idx):
    model, device = models[gpu_id]
    local_embs = []
    total = len(texts_chunk)
    pbar = tqdm(total=total, desc=f"GPU {gpu_id}", unit="chunks", position=gpu_id, leave=False)
    for i in range(0, total, BATCH_SIZE_PER_GPU):
        batch = texts_chunk[i : i + BATCH_SIZE_PER_GPU]
        local_embs.append(embed_chunk(batch, model, device))
        pbar.update(len(batch))
    pbar.close()
    result_list[idx] = np.vstack(local_embs)

# Run embedding
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if NUM_GPUS >= 2:
    split = len(content) // 2
    results = [None, None]
    threads = [
        threading.Thread(target=process_on_gpu, args=(0, content[:split], results, 0)),
        threading.Thread(target=process_on_gpu, args=(1, content[split:], results, 1)),
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    all_embeddings = [results[0], results[1]]
else:
    model, device = models[0]
    all_embeddings = []
    total = len(content)
    pbar = tqdm(total=total, desc="Embedding", unit="chunks")
    for i in range(0, total, BATCH_SIZE_PER_GPU):
        batch = content[i : i + BATCH_SIZE_PER_GPU]
        all_embeddings.append(embed_chunk(batch, model, device))
        pbar.update(len(batch))
    pbar.close()

embeddings = np.vstack(all_embeddings).astype(np.float32)
print(f"Embeddings shape: {embeddings.shape}")
del all_embeddings, df; gc.collect()

# Build FAISS index
print("Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

faiss.write_index(index, os.path.join(VECTOR_STORE_DIR, "faiss_index.bin"))

with open(os.path.join(VECTOR_STORE_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump({"chunk_ids": ids, "metadata": metadata}, f)

with open(os.path.join(VECTOR_STORE_DIR, "config.json"), "w") as f:
    json.dump({
        "model": MODEL_NAME,
        "dim": dim,
        "num_vectors": len(ids),
        "index_type": "IndexFlatIP (cosine)",
        "normalized": True,
        "max_seq_length": MAX_SEQ_LEN,
        "embedding_text_columns": EMBEDDING_TEXT_COLUMNS,
        "embedding_text_version": "metadata_plus_content_v1",
        "data_path": str(DATA_PATH),
        "data_rows": len(ids),
    }, f, indent=2)

for f in os.listdir(CHECKPOINT_DIR):
    os.remove(os.path.join(CHECKPOINT_DIR, f))
os.rmdir(CHECKPOINT_DIR)

print("Output files:")
for f in os.listdir(VECTOR_STORE_DIR):
    fpath = os.path.join(VECTOR_STORE_DIR, f)
    print(f"  {f}: {os.path.getsize(fpath) / 1e9:.2f} GB")
print("Done!")
