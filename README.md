# RAG — Retrieval-Augmented Generation (short)

A compact demo that ingests PDFs, creates embeddings, and uses a vector store + LLM to answer natural language queries using retrieved context.

## Quick Start

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
# or: pip install langchain langchain_groq sentence_transformers chromadb scikit-learn python-dotenv pymupdf
```

3. Add environment variables (optional):
- `GROQ_API_KEY` — for `langchain_groq` integration
- `TOGETHER_KEY`, `PROXY_URL`, `PROXY_HEADERS` — if needed

4. Place PDFs in `data/pdf` and run the notebooks:
- `RAG_intro.ipynb` — quick examples
- `notebook/data_ingestion.ipynb` — ingestion, embedding, and retrieval pipeline

## Notes
- Chromadb persistent store is at `data/vector_store` by default.
- The project uses `SentenceTransformers` for embeddings and a simple retriever for RAG.

---
If you'd like, I can further expand the README with detailed installation steps, examples, or add a `requirements.txt` file.
