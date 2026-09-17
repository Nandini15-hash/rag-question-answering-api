"""
Answer generation, pluggable between OpenAI and a local extractive fallback.

If OPENAI_API_KEY is set, a real chat completion is used with the retrieved
chunks stuffed into the prompt (a standard "stuff" RAG prompt — simple and
transparent, and justified over map-reduce/refine strategies here because
top_k is small (default 4) so the whole context comfortably fits one prompt).

If no key is set, `extractive_fallback` returns the single most relevant
chunk verbatim with light templating instead of failing or mocking a fake
LLM call. This keeps the system runnable end-to-end (and honestly
demoable — the answer really did come from retrieval, not a stub) for
reviewers who don't want to hand out an API key just to test it. The
`generation_mode` field on every response makes it obvious which path
produced a given answer.

NOT-IN-DOCUMENTS HANDLING (required: say so instead of inventing an answer).
Two layers, because a single similarity cutoff can't carry this alone:
  1. A hard floor (`min_similarity_floor`) below which retrieval found
     essentially nothing relevant — every mode short-circuits to an honest
     "not found" answer here, without spending an LLM call.
  2. Above that floor, the OpenAI path defers to the model's own judgment
     (SYSTEM_PROMPT instructs it to say so if the context doesn't answer the
     question) rather than a second hard cutoff — the model can read the
     actual text, a fixed number can't. The extractive fallback has no such
     judgment available, so it uses `similarity_threshold` as its only
     signal. See EXPLANATION.md for why that threshold is unreliable with
     the "offline" embedder specifically (relevant/irrelevant score ranges
     overlap there) and reliable with "local"/"openai".
"""
from openai import APIError, AuthenticationError, RateLimitError

from app.config import settings

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer the user's question "
    "using ONLY the provided context chunks. If the context does not contain the "
    "answer, say so explicitly (e.g. \"The documents don't contain this information.\") "
    "instead of guessing or using outside knowledge. Cite which chunk(s) you used "
    "by their number, e.g. [1]."
)

NOT_FOUND_MESSAGE = (
    "I couldn't find information about that in the uploaded documents. "
    "The closest matches (shown in sources below) don't appear relevant enough to answer this question."
)


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context_block = "\n\n".join(f"[{i + 1}] (from {c['filename']}) {c['text']}" for i, c in enumerate(chunks))
    return f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"


def generate_openai(question: str, chunks: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = _build_prompt(question, chunks)
    try:
        response = client.chat.completions.create(
            model=settings.generation_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            timeout=20,
        )
    except AuthenticationError as exc:
        raise RuntimeError("OpenAI rejected the API key (check OPENAI_API_KEY in .env).") from exc
    except RateLimitError as exc:
        raise RuntimeError("OpenAI rate limit or quota exceeded — try again shortly.") from exc
    except APIError as exc:
        raise RuntimeError(f"OpenAI API error: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — network/timeout and anything else
        raise RuntimeError(f"Could not reach OpenAI: {exc}") from exc

    if not response.choices:
        raise RuntimeError("OpenAI returned an empty response (no choices).")
    return response.choices[0].message.content.strip()


def generate_extractive_fallback(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return NOT_FOUND_MESSAGE
    best = chunks[0]
    if best["similarity_score"] < settings.similarity_threshold:
        return NOT_FOUND_MESSAGE
    preview = best["text"].strip()
    if len(preview) > 500:
        preview = preview[:500].rsplit(" ", 1)[0] + "..."
    return (
        f"(No LLM API key configured — returning the most relevant passage instead of a "
        f"generated answer.)\n\nMost relevant passage, from \"{best['filename']}\" "
        f"[similarity {best['similarity_score']:.3f}]:\n\n{preview}"
    )


def generate_answer(question: str, chunks: list[dict]) -> tuple[str, str]:
    """Returns (answer, generation_mode). generation_mode is "openai",
    "extractive_fallback", or "not_found" (the hard-floor short-circuit)."""
    top_similarity = chunks[0]["similarity_score"] if chunks else 0.0
    if top_similarity < settings.min_similarity_floor:
        return NOT_FOUND_MESSAGE, "not_found"

    if settings.openai_api_key:
        return generate_openai(question, chunks), "openai"
    return generate_extractive_fallback(question, chunks), "extractive_fallback"
