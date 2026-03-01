import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import os
import glob
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

RAW_DIR = os.path.join("data", "raw_docs")
INDEX_DIR = os.path.join("data", "index")
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

def read_all_docs():
    paths = glob.glob(os.path.join(RAW_DIR, "**", "*.txt"), recursive=True) + \
            glob.glob(os.path.join(RAW_DIR, "**", "*.md"), recursive=True)
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            docs.append((p, f.read()))
    return docs

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    docs = read_all_docs()
    if not docs:
        raise RuntimeError(f"No .txt/.md files found in {RAW_DIR}. Put docs into data/raw_docs first.")

    model_name = "all-MiniLM-L6-v2"
    emb_model = SentenceTransformer(model_name)

    chunks = []
    texts = []

    for path, text in docs:
        base = os.path.basename(path)
        for i, ch in enumerate(chunk_text(text)):
            item = {
                "source": base,
                "chunk_id": i,
                "text": ch
            }
            chunks.append(item)
            texts.append(ch)

    embeddings = []
    for i in tqdm(range(0, len(texts), 64), desc="Embedding"):
        batch = texts[i:i+64]
        vecs = emb_model.encode(batch, normalize_embeddings=True)  # cosine-ready
        embeddings.append(vecs)

    emb = np.vstack(embeddings).astype(np.float32)
    np.save(EMB_PATH, emb)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for item in chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Indexed {len(chunks)} chunks.")
    print(f"- Embeddings: {EMB_PATH}")
    print(f"- Chunks:     {CHUNKS_PATH}")

if __name__ == "__main__":
    main()
