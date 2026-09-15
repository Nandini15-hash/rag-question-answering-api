"""
Background job queue for document ingestion.

Why a real queue instead of FastAPI's BackgroundTasks: BackgroundTasks runs
the job on the same event loop after the response is sent, which is fine for
a quick fire-and-forget email, but embedding a large PDF is CPU-bound and can
take several seconds — doing that on the event loop would stall every other
request. Instead, uploads are handed to a small ThreadPoolExecutor worker
pool, and job state (queued/processing/done/failed) is persisted to SQLite
so `/documents/{job_id}/status` reflects real progress and survives a
restart. This is the same shape as a "real" task queue (Celery/RQ + Redis)
at a fraction of the operational cost — justified here because the ingestion
volume for this assignment is small and doesn't need a separate broker
process.
"""
import json
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.embeddings.embedder import get_embedder
from app.ingestion.chunking import chunk_text
from app.ingestion.loaders import load_document
from app.vectorstore.faiss_store import get_store

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    status TEXT NOT NULL,
    chunk_count INTEGER,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class JobStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, document_id: str, filename: str, filepath: str) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, document_id, filename, filepath, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'queued', ?, ?)",
                (job_id, document_id, filename, filepath, now, now),
            )
        return job_id

    def update(self, job_id: str, status: str, chunk_count: int | None = None, error: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, chunk_count=?, error=?, updated_at=? WHERE job_id=?",
                (status, chunk_count, error, now, job_id),
            )

    def get(self, job_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        """Latest job per document_id (a document may have been re-ingested)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs j1 WHERE created_at = ("
                "  SELECT MAX(created_at) FROM jobs j2 WHERE j2.document_id = j1.document_id"
                ") ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]


_job_store: JobStore | None = None


def get_job_store() -> JobStore:
    global _job_store
    if _job_store is None:
        _job_store = JobStore(settings.job_db_path)
    return _job_store


def _log_metric(event: dict) -> None:
    Path(settings.metrics_log_path).parent.mkdir(parents=True, exist_ok=True)
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(settings.metrics_log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


def _process_job(job_id: str) -> None:
    store = get_job_store()
    job = store.get(job_id)
    if job is None:
        return
    store.update(job_id, status="processing")
    t0 = datetime.now(timezone.utc)
    try:
        text = load_document(Path(job["filepath"]))
        chunks = chunk_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        if not chunks:
            raise ValueError("Document produced no extractable text (empty or unreadable file).")

        embedder = get_embedder()
        vectors = embedder.embed([c.text for c in chunks])

        vector_store = get_store(settings.index_dir, embedder.dimension)
        metadatas = [
            {
                "document_id": job["document_id"],
                "filename": job["filename"],
                "chunk_index": c.index,
                "text": c.text,
            }
            for c in chunks
        ]
        vector_store.add(vectors, metadatas)

        store.update(job_id, status="done", chunk_count=len(chunks))
        elapsed_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000
        _log_metric(
            {
                "event": "ingestion",
                "document_id": job["document_id"],
                "filename": job["filename"],
                "chunk_count": len(chunks),
                "duration_ms": round(elapsed_ms, 1),
            }
        )
    except Exception as exc:  # noqa: BLE001 — surface any failure onto the job record
        store.update(job_id, status="failed", error=str(exc))
        _log_metric({"event": "ingestion_failed", "document_id": job["document_id"], "error": str(exc)})


class IngestionWorkerPool:
    def __init__(self, max_workers: int):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest")

    def submit(self, job_id: str) -> None:
        self._executor.submit(_process_job, job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


_worker_pool: IngestionWorkerPool | None = None


def get_worker_pool() -> IngestionWorkerPool:
    global _worker_pool
    if _worker_pool is None:
        _worker_pool = IngestionWorkerPool(settings.ingestion_workers)
    return _worker_pool
