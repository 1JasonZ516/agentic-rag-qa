import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import os
import sys
import json
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.join("data", "index")
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.jsonl")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

SYSTEM_RULES = """You are a careful assistant.
Use ONLY the provided context to answer.
For every factual statement, append citations in the form [source#chunk_id].
If the context is insufficient, say: "I don't have enough evidence in the provided documents."
Then list the top relevant chunks as bullet points with their [source#chunk_id].
"""

def ollama_generate(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def load_index():
    if not (os.path.exists(EMB_PATH) and os.path.exists(CHUNKS_PATH)):
        raise RuntimeError("Index not found. Run: python src\\ingest.py")

    emb = np.load(EMB_PATH)
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return emb, chunks

def build_prompt(question: str, contexts):
    blocks = []
    for item in contexts:
        source = item["source"]
        cid = item["chunk_id"]
        text = item["text"]
        blocks.append(f"[{source}#{cid}]\n{text}")
    ctx = "\n\n".join(blocks)

    return f"""{SYSTEM_RULES}

Context:
{ctx}

Question: {question}
Answer:"""

def main():
    if len(sys.argv) < 2:
        print('Usage: python src\\ask.py "your question"')
        sys.exit(1)

    question = sys.argv[1]

    emb, chunks = load_index()

    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    q = emb_model.encode([question], normalize_embeddings=True).astype(np.float32)[0]

    # cosine similarity since embeddings are normalized: dot product
    sims = emb @ q
    topk = int(min(4, len(sims)))
    idx = np.argsort(-sims)[:topk]

    contexts = [chunks[i] for i in idx]
    prompt = build_prompt(question, contexts)
    answer = ollama_generate(prompt)

    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== Retrieved Chunks ===\n")
    for item in contexts:
        print(f"- [{item['source']}#{item['chunk_id']}] (score={float(sims[chunks.index(item)]):.3f})")

if __name__ == "__main__":
    main()
