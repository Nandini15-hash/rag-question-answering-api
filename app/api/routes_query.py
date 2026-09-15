import time

from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.generation.llm import generate_answer
from app.middleware.rate_limit import limiter
from app.models import QueryMetrics, QueryRequest, QueryResponse, RetrievedChunk
from app.retrieval.retriever import retrieve
from app.utils.metrics import log_query_metric

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
@limiter.limit(settings.rate_limit)
async def query(request: Request, body: QueryRequest):
    try:
        chunks, retrieval_latency_ms = retrieve(body.question, top_k=body.top_k, document_ids=body.document_ids)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Retrieval failed: {exc}") from exc

    if not chunks:
        raise HTTPException(
            404,
            "No indexed documents found. Upload at least one document via /documents/upload and wait for its "
            "ingestion job to complete before querying.",
        )

    gen_t0 = time.perf_counter()
    try:
        answer, generation_mode = generate_answer(body.question, chunks)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Answer generation failed: {exc}") from exc
    generation_latency_ms = (time.perf_counter() - gen_t0) * 1000

    top_similarity = chunks[0]["similarity_score"] if chunks else None
    log_query_metric(
        question=body.question,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        top_similarity_score=top_similarity,
        generation_mode=generation_mode,
        result_count=len(chunks),
    )

    return QueryResponse(
        answer=answer,
        sources=[RetrievedChunk(**c) for c in chunks],
        metrics=QueryMetrics(
            retrieval_latency_ms=round(retrieval_latency_ms, 1),
            generation_latency_ms=round(generation_latency_ms, 1),
            total_latency_ms=round(retrieval_latency_ms + generation_latency_ms, 1),
            top_similarity_score=top_similarity,
            generation_mode=generation_mode,
        ),
    )
