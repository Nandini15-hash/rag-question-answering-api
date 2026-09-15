# Explanation

This document covers the three things the assignment asks to be explained
explicitly: chunk size choice, a retrieval failure actually observed, and a
metric tracked. All numbers below are real output from running this system
end-to-end against the two sample documents in `sample_docs/` (a plaintext
employee handbook and a generated PDF product overview) — commands to
reproduce them are in the last section.

## 1. Why this chunk size

`CHUNK_SIZE=800` characters with `CHUNK_OVERLAP=120` (both configurable via
`.env`).

Reasoning, in order of how much each factor weighed in:

- **The documents are policy/reference text with short, self-contained
  paragraphs** (a handbook section, a product-doc subsection), typically
  100–500 characters each. 800 characters is roughly 1–3 paragraphs, which
  keeps a chunk topically coherent — one chunk is normally "about" one
  thing (leave policy, or pricing), which is exactly the unit a similarity
  search wants to match against a question that's also "about" one thing.
  A much larger chunk (e.g. 2000+ chars) would frequently mix two unrelated
  topics into one embedding, diluting the vector and hurting precision.
- **Smaller chunks (e.g. 200 chars) would fragment sentences and cross-
  references.** Several answers in the handbook depend on two adjacent
  sentences (e.g. "accrue 1.5 days... capped at 18 days... unused leave
  beyond 5 days does not carry over" — three related facts in one Leave
  Policy paragraph). Splitting these across three tiny chunks means no
  single chunk contains the full answer, and retrieval would need to
  return all three chunks correctly ranked near the top — much harder than
  needing one well-formed chunk.
- **800 characters is comfortably inside the context window of any modern
  embedding model or LLM**, so there's no truncation risk, and with
  `top_k=4` (default) the full retrieved context is at most ~3200
  characters — small enough that "stuffing" all of it into one LLM prompt
  (rather than a map-reduce/refine strategy) is simple and cheap, which is
  the generation strategy this project uses (see `app/generation/llm.py`).
- **120-character overlap (15% of chunk size)** exists specifically so a
  fact sitting near a chunk boundary is still retrievable from whichever
  side of the boundary the query's phrasing happens to match — see the
  chunking module's docstring (`app/ingestion/chunking.py`) for the actual
  splitting algorithm (paragraph → sentence → word → hard cut, in that
  order, so boundaries land on natural language breaks whenever possible
  rather than mid-word).

These are reasonable defaults for short-form reference documents, not a
universal answer — a corpus of long narrative text (research papers, books)
would likely want a larger chunk size, and highly structured data (tables,
logs) would want a completely different, structure-aware splitter.

## 2. A retrieval failure actually observed

Setup: both sample documents ingested (`company_handbook.txt`,
`forge_product_overview.pdf`), then queried via `/query`.

**Query:** *"What is the deadline for submitting an expense reimbursement
claim?"*

**Expected top result:** the Expense Reimbursement chunk from
`company_handbook.txt`, which literally says *"All claims must be submitted
through the finance portal within 30 days of the expense being incurred."*

**What actually happened** (this run used `EMBEDDING_PROVIDER=offline`, the
hashing-based fallback — see below for why):

| rank | source | similarity |
|---|---|---|
| 1 | `forge_product_overview.pdf` chunk 2 (Pricing/Security section) | 0.275 |
| 2 | `company_handbook.txt` chunk 2 (**the correct chunk**) | 0.235 |
| 3 | `forge_product_overview.pdf` chunk 1 | 0.231 |
| 4 | `forge_product_overview.pdf` chunk 0 | 0.212 |

The correct chunk was retrieved (it made the top-4), but it was **not
ranked first** — an unrelated pricing/compliance chunk from a completely
different document scored higher. Because the current prompt/fallback logic
leans on the single top-ranked chunk (the extractive fallback answers from
`sources[0]` specifically), this would have produced a wrong or irrelevant
answer if `top_k` were set to 1.

**Root cause:** this specific failure is a property of the *offline hashing
embedder*, not of the retrieval pipeline in general. `HashingEmbedder`
(`app/embeddings/embedder.py`) is a bag-of-words representation — it has no
notion that "deadline" and "within 30 days" are semantically related, or
that "submitting a claim" and "claims must be submitted" are the same idea
in different word order. It only measures token/token-pair overlap (with
hash collisions on top, since it maps into a fixed 512-dim space). The
Forge chunk happened to share enough incidental tokens (structural/
formatting words, similar sentence lengths) to edge out the semantically
correct chunk.

Note on how this was tested: the sandbox this project was built in has an
outbound network allowlist that blocks `huggingface.co` (confirmed via a
403 policy denial on the model download, not a transient failure), so the
`local` sentence-transformers provider could not actually be downloaded
there to re-run this exact comparison. The failure above was reproduced
with `EMBEDDING_PROVIDER=offline` specifically because it needs no network
call at all. On a normal machine with internet access (i.e. wherever this
is actually reviewed/run), re-run the reproduction steps below with
`EMBEDDING_PROVIDER=local` (the shipped default) — a trained embedding
model encoding "deadline" and "within 30 days" as semantically close should
rank the correct handbook chunk first. **This is exactly why the offline
hashing provider is documented as a fallback for locked-down environments
only, and `local`/`openai` are the recommended defaults** — this failure
case is a concrete illustration of the bag-of-words quality gap, not a bug
to fix in the retrieval code itself.

A fix that *would* help regardless of embedding provider: retrieve a larger
`top_k` and let the LLM (not just the top-1 chunk) decide which passage
actually answers the question, rather than the extractive fallback's
top-1-only heuristic — which is what happens automatically once
`OPENAI_API_KEY` is set, since the generation prompt includes all `top_k`
chunks, not just the first.

## 3. Metric tracked

Every `/query` call logs a structured JSON line to
`data/metrics.jsonl` (path configurable via `METRICS_LOG_PATH`) via
`app/utils/metrics.py`. The fields tracked are:

- `retrieval_latency_ms` — time from receiving the question to having
  ranked FAISS results (embed query + similarity search).
- `generation_latency_ms` — time spent in the LLM call (or the
  fallback's negligible string formatting).
- `total_latency_ms` — sum of the two, i.e. end-to-end API latency for
  the query beyond FastAPI's own request handling.
- `top_similarity_score` — the cosine similarity of the single best-
  matching chunk, used as a proxy for "how confident is this answer".
- `generation_mode` — `openai` or `extractive_fallback`, so latency and
  quality can be compared across the two paths.

**Why these, and what they showed in practice:** `retrieval_latency_ms`
is the one I paid closest attention to, because it's the number that
scales with corpus size and directly bounds API responsiveness. Across the
six real queries run against this two-document corpus (visible in
`data/metrics.jsonl` after running the reproduction steps below), retrieval
consistently took under 1ms (FAISS `IndexFlatIP` brute-force search over a
handful of vectors is essentially free) — the real cost, once
`OPENAI_API_KEY` is set, is entirely the LLM call, not retrieval. That's
useful operationally: it tells you that scaling this system to thousands of
documents will not meaningfully change query latency until the corpus is
large enough that exact FAISS search itself becomes the bottleneck (tens to
hundreds of thousands of vectors), at which point swapping `IndexFlatIP`
for an approximate index (`IndexIVFFlat`/HNSW) is the natural next step —
and `top_similarity_score` is the metric that would tell you whether that
switch (which trades exact recall for speed) is degrading result quality.

## Reproducing these numbers

```bash
source .venv/bin/activate
export EMBEDDING_PROVIDER=offline   # or "local" with internet access, for comparison
uvicorn app.main:app --port 8000 &

curl -X POST http://127.0.0.1:8000/documents/upload -F "file=@sample_docs/company_handbook.txt"
curl -X POST http://127.0.0.1:8000/documents/upload -F "file=@sample_docs/forge_product_overview.pdf"
# poll /documents/{job_id}/status until both are "done", then:

curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What is the deadline for submitting an expense reimbursement claim?"}'

cat data/metrics.jsonl
```
