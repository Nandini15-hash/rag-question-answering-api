"""Pydantic request/response schemas — the "Request validation" requirement.

Every endpoint that takes a body or returns structured data is typed here so
FastAPI validates input automatically and generates accurate OpenAPI docs.
"""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class UploadResponse(BaseModel):
    job_id: str
    document_id: str
    filename: str
    status: JobStatus
    message: str = "Document accepted for background ingestion."


class JobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    filename: str
    status: JobStatus
    chunk_count: int | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    status: JobStatus
    chunk_count: int | None = None
    created_at: datetime


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these document IDs. Omit to search all documents."
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class RetrievedChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    similarity_score: float


class QueryMetrics(BaseModel):
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    top_similarity_score: float | None = None
    generation_mode: str  # "openai" | "extractive_fallback" | "not_found"


class QueryResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    metrics: QueryMetrics


class ErrorResponse(BaseModel):
    detail: str
