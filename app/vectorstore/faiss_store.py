"""
Local FAISS vector store with a sidecar JSON metadata file.

Why FAISS over a hosted store like Pinecone: for an assignment-scale corpus
(a handful of documents, thousands of chunks at most) an in-process
IndexFlatIP is exact (no ANN recall loss), free, and has zero network
latency — Pinecone's value (managed scaling, multi-node ANN) doesn't pay for
itself at this scale and would add an external dependency + API key just to
stand the project up. IndexFlatIP does brute-force cosine similarity (since
embeddings are L2-normalized, inner product == cosine similarity), which is
fine up to ~100k vectors on CPU.

The index and its metadata (chunk text, document id, filename, chunk index)
are persisted to disk so ingested documents survive a server restart.
"""
import json
import threading
from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    def __init__(self, index_dir: str, dimension: int):
        self._dir = Path(index_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.faiss"
        self._meta_path = self._dir / "metadata.json"
        self._dimension = dimension
        self._lock = threading.Lock()

        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            self._metadata: list[dict] = json.loads(self._meta_path.read_text())
        else:
            self._index = faiss.IndexFlatIP(dimension)
            self._metadata = []

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        assert vectors.shape[0] == len(metadatas)
        assert vectors.shape[1] == self._dimension, (
            f"Embedding dimension {vectors.shape[1]} does not match index dimension "
            f"{self._dimension}. Did EMBEDDING_PROVIDER change after the index was built? "
            f"Delete {self._dir} to rebuild from scratch."
        )
        with self._lock:
            self._index.add(vectors)
            self._metadata.extend(metadatas)
            self._persist()

    def search(self, query_vector: np.ndarray, k: int, document_ids: list[str] | None = None) -> list[dict]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            # over-fetch when filtering by document_ids since IndexFlatIP has no native filter
            fetch_k = min(self._index.ntotal, k * 5 if document_ids else k)
            scores, indices = self._index.search(query_vector.reshape(1, -1), fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            if document_ids and meta["document_id"] not in document_ids:
                continue
            results.append({**meta, "similarity_score": float(score)})
            if len(results) >= k:
                break
        return results

    def document_chunk_count(self, document_id: str) -> int:
        return sum(1 for m in self._metadata if m["document_id"] == document_id)

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(json.dumps(self._metadata))


_store: FaissStore | None = None
_store_lock = threading.Lock()


def get_store(index_dir: str, dimension: int) -> FaissStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = FaissStore(index_dir, dimension)
        return _store
