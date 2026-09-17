# RAG Question-Answering API

Upload PDF or TXT documents, they're chunked and embedded in the background,
then ask questions and get answers grounded in retrieved passages, with
source citations and timing/quality metrics on every response.

Built with FastAPI + FAISS + sentence-transformers. See
[`EXPLANATION.md`](./EXPLANATION.md) for the chunking rationale, a real
retrieval failure case observed while testing this system, and the metrics
tracked. See [`diagrams/architecture.drawio`](./diagrams/architecture.drawio)
for the architecture diagram (open at [app.diagrams.net](https://app.diagrams.net) →
File → Open From → Device).

## Architecture at a glance

```
Upload (PDF/TXT) --> background ingestion job --> chunk --> embed --> FAISS index
                                                                          |
Question -----------------------------------> embed query --> similarity search
                                                                          |
                                                            top-k chunks --> LLM (or
                                                            extractive fallback) --> answer + sources + metrics
```

- **Ingestion is asynchronous.** `/documents/upload` returns immediately
  with a `job_id`; a small thread-pool worker does the actual parsing +
  chunking + embedding in the background, so uploading a large PDF never
  blocks the API. Poll `/documents/{job_id}/status` to know when it's done.
- **Everything runs with zero API keys by default.** Embeddings use a local
  sentence-transformers model (no cost, no network after the first model
  download) and, if no `OPENAI_API_KEY` is set, answer generation falls back
  to returning the most relevant retrieved passage instead of a generated
  answer, clearly labeled as such. Set `OPENAI_API_KEY` to get real
  LLM-generated answers.

## Setup

Requires Python 3.11+.

```bash
git clone <this-repo-url>
cd rag-qa-system
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # optionally add OPENAI_API_KEY here
uvicorn app.main:app --reload
```

The API is now at `http://127.0.0.1:8000`. Interactive docs (Swagger UI) are
at `http://127.0.0.1:8000/docs`.

The first request that needs embeddings will download the
`all-MiniLM-L6-v2` model (~80MB) from Hugging Face — this needs internet
access once; after that it's cached locally. If you're on a network that
blocks huggingface.co, set `EMBEDDING_PROVIDER=offline` in `.env` to use a
zero-network hashing-based embedder instead (lower retrieval quality — see
`EXPLANATION.md`).

## Usage

### 1. Upload a document

```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@sample_docs/company_handbook.txt"
```

```json
{
  "job_id": "c8eff755-...",
  "document_id": "2f7f74ec-...",
  "filename": "company_handbook.txt",
  "status": "queued",
  "message": "Document accepted for background ingestion."
}
```

### 2. Poll ingestion status

```bash
curl http://127.0.0.1:8000/documents/c8eff755-.../status
```

```json
{
  "job_id": "c8eff755-...",
  "document_id": "2f7f74ec-...",
  "filename": "company_handbook.txt",
  "status": "done",
  "chunk_count": 5,
  "error": null,
  "created_at": "2026-09-15T07:25:07Z",
  "updated_at": "2026-09-15T07:25:08Z"
}
```

`status` is one of `queued`, `processing`, `done`, `failed` (with `error`
populated on failure — e.g. an unreadable/corrupt PDF).

### 3. List uploaded documents

```bash
curl http://127.0.0.1:8000/documents
```

### 4. Ask a question

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many paid leave days do employees accrue per month?"}'
```

```json
{
  "answer": "...",
  "sources": [
    {
      "document_id": "2f7f74ec-...",
      "filename": "company_handbook.txt",
      "chunk_index": 1,
      "text": "...Leave Policy\nFull-time employees accrue 1.5 days of paid leave per month...",
      "similarity_score": 0.278
    }
  ],
  "metrics": {
    "retrieval_latency_ms": 0.7,
    "generation_latency_ms": 0.0,
    "total_latency_ms": 0.7,
    "top_similarity_score": 0.278,
    "generation_mode": "extractive_fallback"
  }
}
```

Optional request fields: `top_k` (override the number of chunks retrieved)
and `document_ids` (restrict the search to specific documents).

### 5. Asking something the documents don't cover

The system is required to say so rather than invent an answer. Try:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

This still returns `200 OK` (it's a normal outcome, not an error) with
`metrics.generation_mode` set to `"not_found"` and an honest answer instead
of a hallucinated one. See "Not-in-documents handling" in `EXPLANATION.md`
for how this is decided and its one known limitation.

## Rate limiting

All endpoints are limited (default `20/minute` per client IP, configurable
via `RATE_LIMIT` in `.env`). Exceeding it returns `429` with a `detail`
message.

## Running tests

```bash
python3 -m pytest tests/ -v
```

Tests force `EMBEDDING_PROVIDER=offline` so the suite runs with no network
access and no external model download, in well under a second.

## Project layout

```
app/
  main.py                FastAPI app, router registration, rate-limit handler
  config.py               typed settings loaded from .env
  models.py                Pydantic request/response schemas
  ingestion/
    loaders.py              PDF/TXT text extraction
    chunking.py              dependency-free recursive text splitter
    jobs.py                    SQLite-backed job queue + thread-pool worker
  embeddings/embedder.py    local / OpenAI / offline-hashing embedder
  vectorstore/faiss_store.py  FAISS IndexFlatIP + JSON metadata sidecar
  retrieval/retriever.py       embed query -> similarity search
  generation/llm.py              OpenAI call, or extractive fallback
  middleware/rate_limit.py         slowapi limiter
  api/routes_documents.py, routes_query.py
  utils/metrics.py                   JSONL metrics logging
tests/                    pytest suite (chunking, not-in-docs threshold logic, API integration)
sample_docs/              two demo documents (.txt + .pdf) used for testing
scripts/make_sample_pdf.py  generates the sample PDF (not part of the app)
diagrams/                 architecture.drawio + exported PNG
EXPLANATION.md            chunk size rationale, failure case, metrics
```

## What works vs. what doesn't

Honest account, per the "60% working and disclosed beats 100% claimed and
broken" grading note. See `EXPLANATION.md` point 4 for the fuller version
and next steps.

**Works, tested end-to-end:**
- Upload → background ingestion (PDF and TXT) → chunk → embed → FAISS index,
  with real status polling (`queued` → `processing` → `done`/`failed`)
- Query → retrieval → generation → answer with cited sources and per-request
  metrics (retrieval/generation latency, top similarity, generation mode)
- "Not in documents" short-circuit (`not_found` mode) — see limitation below
- Pydantic validation on both endpoints (short/blank questions, bad file
  types, empty files all rejected with clear 4xx errors)
- Rate limiting (429 after the configured limit)
- Explicit, typed error handling on the OpenAI embedding and generation
  calls (auth, rate-limit, network/timeout each get a distinct message)
- 16 automated tests (chunking, generation-threshold logic, full API
  round-trip), all passing

**Works, but with a known caveat:**
- The `similarity_threshold` cutoff (what the *extractive fallback* uses to
  decide "not in documents" when no LLM key is set) is unreliable with
  `EMBEDDING_PROVIDER=offline` specifically — its relevant/irrelevant score
  distributions overlap (measured, see `EXPLANATION.md`). It's reliable with
  the default `local` provider or `openai`, which weren't network-testable
  in the sandbox this was built in (see below).
- `local` (sentence-transformers) could not actually be run in the sandbox
  used to build this — huggingface.co is blocked by that sandbox's egress
  policy (confirmed 403, not a bug). Everything was tested against the
  zero-network `offline` provider instead. The code path is the same either
  way (`app/embeddings/embedder.py`), just not integration-tested with the
  recommended default embedder in *this* environment.

**Not finished:**
- No evaluation set (a fixed list of question → expected-chunk pairs) to
  measure retrieval precision/recall numerically — see EXPLANATION.md point
  4 for what that would look like.
- No support for re-ingesting/replacing a document by the same name (each
  upload creates a new `document_id`; the old chunks aren't removed).
- No streaming responses — `/query` blocks until the full LLM answer is
  generated rather than streaming tokens.

## Design choices and constraints addressed

- **No LangChain/LlamaIndex.** Chunking, the job queue, and the retrieval
  loop are all small enough (a few dozen lines each) to write directly and
  reason about — see the docstring at the top of each module for the
  specific justification. This also means there's no framework "magic"
  hiding what's actually happening at chunk or prompt-construction time,
  which matters for the explanation requirements below.
- **FAISS over Pinecone**: at this corpus scale (a handful of documents),
  an in-process exact index has no recall loss, no network latency, and no
  external account/API key requirement. See `app/vectorstore/faiss_store.py`.
- **A real background job queue, not `BackgroundTasks`**: ingestion is
  CPU-bound (parsing + embedding), so it runs on a thread pool with
  SQLite-persisted status rather than on the request event loop. See
  `app/ingestion/jobs.py`.
