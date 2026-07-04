# Local RAG Document Q&A

A Retrieval-Augmented Generation (RAG) app that answers questions about an uploaded PDF, grounded strictly in the document's content. Built end-to-end: chunking, embedding, vector retrieval, and LLM generation, with a Streamlit interface and a full evaluation pass documenting where naive RAG holds up and where it breaks.

**Live demo:** https://huggingface.co/spaces/sam2103/rag-for-pdfs
*(Free-tier hosting — the app may take a moment to wake up if it's been inactive.)*

## What it does

1. Upload a PDF through the browser
2. The document is chunked, embedded, and stored in a local vector database
3. Ask a question — the app retrieves the most relevant chunks and generates an answer using only that retrieved context
4. Every answer shows its source chunks, so you can verify what it was grounded in

## Tech stack

| Component | Tool |
|---|---|
| Frontend | Streamlit |
| Chunking | Custom (1000 chars, 200-char overlap) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent) |
| Generation (local dev) | Llama 3 8B via Ollama |
| Generation (deployed) | `openai/gpt-oss-20b` via Groq API |
| Deployment | Docker, hosted on Hugging Face Spaces |

Generation is dual-mode: local development runs entirely offline through Ollama; the deployed version swaps to Groq's hosted API, since free-hosting tiers can't run a local LLM runtime. Retrieval and embeddings run identically in both environments.

## Running locally

```bash
git clone <this-repo>
cd rag-doc-qa
pip install -r requirements.txt
```

Create a `.env` file:
```
LLM_PROVIDER=ollama
```
(For local generation, install [Ollama](https://ollama.com) and pull `llama3`. To test the Groq path locally instead, set `LLM_PROVIDER=groq` and add `GROQ_API_KEY` / `GROQ_MODEL`.)

Run the app:
```bash
streamlit run app.py
```

## Running with Docker

```bash
docker build -t rag-doc-qa .
docker run -p 8501:8501 --env-file .env rag-doc-qa
```

## Evaluation & known limitations

This project includes a documented evaluation pass, not just a working demo. Testing across five query categories (direct fact, paraphrase, specific figures, multi-hop, out-of-scope) found:

- Retrieval reliably handles single-topic queries, including genuine paraphrases with no lexical overlap with the source text
- **Multi-hop questions expose a real limitation**: a single query embedding can't represent two distinct topics, so naive top-k retrieval silently drops half the answer
- **Larger models can silently violate grounding instructions**, answering from pretrained knowledge rather than retrieved context, in a way that's undetectable without manually cross-checking sources
- Locally-quantized models showed non-determinism even at `temperature=0` with a fixed seed

Full findings, methodology, and attempted fixes (including query decomposition) are in [`EVAL.md`](./EVAL.md).

## Project structure

```
├── app.py              # Streamlit UI, file upload, chat interface
├── ingest.py            # PDF loading, chunking, embedding
├── retrieve.py           # Query embedding + ChromaDB retrieval
├── generate.py           # Prompt construction + dual-mode LLM generation
├── decompose.py          # Query decomposition (experimental, see EVAL.md)
├── retrieve_multi.py     # Multi-query retrieval using decomposition
├── requirements.txt
├── Dockerfile
└── EVAL.md              # Full evaluation writeup
```

## Future work

- Relevance filtering before generation, to catch chunks that are topically related but don't actually answer the question
- Per-claim source attribution in generated answers, to make grounding violations automatically detectable
- Persistent, content-hashed document caching across sessions (implemented for local dev; not yet wired into the deployed version's storage)
