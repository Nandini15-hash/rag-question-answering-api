"""Embed the query, search FAISS, return the top-k chunks with scores."""
import time

from app.embeddings.embedder import get_embedder
from app.vectorstore.faiss_store import get_store
from app.config import settings


def retrieve(question: str, top_k: int | None = None, document_ids: list[str] | None = None):
    t0 = time.perf_counter()
    embedder = get_embedder()
    query_vector = embedder.embed([question])[0]

    store = get_store(settings.index_dir, embedder.dimension)
    results = store.search(query_vector, k=top_k or settings.top_k, document_ids=document_ids)

    latency_ms = (time.perf_counter() - t0) * 1000
    return results, latency_ms
