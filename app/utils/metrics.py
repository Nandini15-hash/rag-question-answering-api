"""Append-only JSONL metrics log — the "track one metric" requirement.

We log more than one metric in practice (retrieval latency, generation
latency, top similarity score) because they were cheap to capture together
and the explanation doc discusses top similarity score and retrieval latency
specifically. Kept as flat JSONL rather than a database so it's trivial to
`tail -f` during a demo or load into pandas for a quick plot.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


def log_query_metric(
    question: str,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    top_similarity_score: float | None,
    generation_mode: str,
    result_count: int,
) -> None:
    Path(settings.metrics_log_path).parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event": "query",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "retrieval_latency_ms": round(retrieval_latency_ms, 1),
        "generation_latency_ms": round(generation_latency_ms, 1),
        "total_latency_ms": round(retrieval_latency_ms + generation_latency_ms, 1),
        "top_similarity_score": round(top_similarity_score, 4) if top_similarity_score is not None else None,
        "generation_mode": generation_mode,
        "result_count": result_count,
    }
    with open(settings.metrics_log_path, "a") as f:
        f.write(json.dumps(event) + "\n")
