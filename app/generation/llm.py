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
"""
from app.config import settings

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer the user's question "
    "using ONLY the provided context chunks. If the context does not contain the "
    "answer, say so explicitly instead of guessing. Cite which chunk(s) you used "
    "by their number, e.g. [1]."
)


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context_block = "\n\n".join(f"[{i + 1}] (from {c['filename']}) {c['text']}" for i, c in enumerate(chunks))
    return f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"


def generate_openai(question: str, chunks: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = _build_prompt(question, chunks)
    response = client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def generate_extractive_fallback(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I couldn't find any relevant content in the uploaded documents to answer that question."
    best = chunks[0]
    preview = best["text"].strip()
    if len(preview) > 500:
        preview = preview[:500].rsplit(" ", 1)[0] + "..."
    return (
        f"(No LLM API key configured — returning the most relevant passage instead of a "
        f"generated answer.)\n\nMost relevant passage, from \"{best['filename']}\" "
        f"[similarity {best['similarity_score']:.3f}]:\n\n{preview}"
    )


def generate_answer(question: str, chunks: list[dict]) -> tuple[str, str]:
    """Returns (answer, generation_mode)."""
    if settings.openai_api_key:
        return generate_openai(question, chunks), "openai"
    return generate_extractive_fallback(question, chunks), "extractive_fallback"
