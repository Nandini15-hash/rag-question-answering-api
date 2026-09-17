"""
Centralized configuration, loaded from environment variables / .env.

Why pydantic-settings: it validates types at startup (e.g. CHUNK_SIZE must be
an int) and gives every other module a single typed `settings` object instead
of scattering `os.getenv()` calls around the codebase.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM / generation
    openai_api_key: str | None = None
    generation_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_provider: str = "local"  # "local" | "openai"
    local_embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 4

    # "Answer not in documents" handling. Two thresholds, not one:
    #  - min_similarity_floor: below this, retrieval found essentially nothing relevant
    #    (near-zero lexical/semantic overlap) — short-circuit to "not found" for every
    #    generation mode, without spending an LLM call on it.
    #  - similarity_threshold: the fallback (no-LLM) generator's only way to judge
    #    relevance is this cosine-similarity cutoff, since it can't reason about the
    #    text. Default (0.16) is calibrated from real offline-embedder measurements
    #    (see EXPLANATION.md) to sit just below the lowest genuinely-relevant score
    #    observed, so relevant queries keep working — the honest tradeoff is that
    #    some irrelevant queries also clear 0.16 and slip through undetected, because
    #    the "offline" hashing embedder's relevant/irrelevant score ranges genuinely
    #    overlap (a real, documented limitation, not tuned away). Raise this
    #    (e.g. ~0.35) if using "local"/"openai" embeddings, whose semantic scores
    #    separate relevant from irrelevant far more cleanly.
    #    The OpenAI generation path does NOT hard-gate on this — it trusts the LLM's
    #    own judgment (instructed via SYSTEM_PROMPT) to say "not in the documents"
    #    when the retrieved chunks don't actually answer the question, which is more
    #    reliable than a fixed cutoff once an LLM is available.
    min_similarity_floor: float = 0.05
    similarity_threshold: float = 0.16

    # Rate limiting (slowapi syntax, e.g. "20/minute")
    rate_limit: str = "20/minute"

    # Storage
    upload_dir: str = "data/uploads"
    index_dir: str = "data/index"
    job_db_path: str = "data/jobs.db"
    metrics_log_path: str = "data/metrics.jsonl"

    # Ingestion background workers
    ingestion_workers: int = 2


settings = Settings()
