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
