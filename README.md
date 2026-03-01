# Local RAG QA with Citations (Ollama + Sentence-Transformers)

A minimal Retrieval-Augmented Generation (RAG) demo that answers questions using only a local document corpus and produces citation-grounded responses in the format `[source#chunk_id]`.

- **LLM:** Ollama (local) ！ `llama3.2:3b`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Retrieval:** local cosine-similarity search over stored embeddings (NumPy)
- **Output:** answers + citations + retrieved evidence chunks

## Project Structure
- `data/raw_docs/` ！ input documents (`.md` / `.txt`)
- `data/index/` ！ generated index files (`embeddings.npy`, `chunks.jsonl`) *(not committed)*
- `src/ingest.py` ！ chunk documents + build embeddings index
- `src/ask.py` ！ retrieve top-k chunks + call local LLM + produce cited answer

## Setup

### 1) Install Ollama and pull model
```powershell
ollama pull llama3.2:3b