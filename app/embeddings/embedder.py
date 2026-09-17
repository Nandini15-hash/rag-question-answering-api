"""
Embedding generation, pluggable between a local model, OpenAI, and an
offline hashing fallback.

Default is "local" (sentence-transformers/all-MiniLM-L6-v2): it runs on CPU,
needs no API key, and its 384-dim vectors are more than good enough for
semantic search over a handful of documents. This means the whole system is
demoable with zero external dependencies or cost. Setting
EMBEDDING_PROVIDER=openai switches to text-embedding-3-small (1536-dim) for
higher-quality embeddings when a key is available — the FAISS index records
its own dimension so the two are never silently mixed.

A third provider, "offline", exists for environments that cannot reach
huggingface.co at all (e.g. a locked-down CI runner or sandbox with an
egress allowlist) and therefore can't even download all-MiniLM-L6-v2 once.
It uses scikit-learn's HashingVectorizer — TF-weighted word hashing into a
fixed-dimension space — which needs no network call ever, at the cost of
being a purely lexical (bag-of-words) representation rather than a learned
semantic one. It's not what we'd recommend for production quality, but it
keeps the pipeline runnable end-to-end anywhere, and its lexical-overlap
behavior is what surfaced the retrieval failure case documented in
EXPLANATION.md.
"""
from functools import lru_cache

import numpy as np

from app.config import settings


class Embedder:
    provider: str
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.provider = "local"
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name: str, api_key: str):
        from openai import OpenAI

        self.provider = "openai"
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        # text-embedding-3-small is 1536-dim; avoid a network call just to learn this.
        self.dimension = 1536

    def embed(self, texts: list[str]) -> np.ndarray:
        from openai import APIError, AuthenticationError, RateLimitError

        try:
            response = self._client.embeddings.create(model=self._model_name, input=texts, timeout=20)
        except AuthenticationError as exc:
            raise RuntimeError("OpenAI rejected the API key while embedding (check OPENAI_API_KEY in .env).") from exc
        except RateLimitError as exc:
            raise RuntimeError("OpenAI rate limit or quota exceeded while embedding — try again shortly.") from exc
        except APIError as exc:
            raise RuntimeError(f"OpenAI API error while embedding: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — network/timeout and anything else
            raise RuntimeError(f"Could not reach OpenAI while embedding: {exc}") from exc

        vectors = [item.embedding for item in response.data]
        arr = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class HashingEmbedder(Embedder):
    """Zero-network fallback: TF-weighted word hashing into a fixed-dim space."""

    def __init__(self, n_features: int = 512):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.provider = "offline"
        self.dimension = n_features
        self._vectorizer = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm=None, ngram_range=(1, 2)
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if settings.embedding_provider == "openai" and settings.openai_api_key:
        return OpenAIEmbedder(settings.openai_embedding_model, settings.openai_api_key)
    if settings.embedding_provider == "offline":
        return HashingEmbedder()
    return LocalEmbedder(settings.local_embedding_model)
