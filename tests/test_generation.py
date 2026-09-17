"""
Unit tests for the "not in documents" handling, using synthetic chunk data
with controlled similarity scores (rather than real embeddings) so the
threshold logic itself is tested deterministically, independent of which
embedding provider is configured.
"""
from app.generation.llm import NOT_FOUND_MESSAGE, generate_answer, generate_extractive_fallback


def _chunk(similarity: float, text: str = "Some retrieved passage.") -> dict:
    return {
        "document_id": "doc1",
        "filename": "test.txt",
        "chunk_index": 0,
        "text": text,
        "similarity_score": similarity,
    }


def test_no_chunks_returns_not_found():
    answer, mode = generate_answer("irrelevant question", [])
    assert mode == "not_found"
    assert answer == NOT_FOUND_MESSAGE


def test_below_floor_short_circuits_regardless_of_llm_key():
    # similarity below min_similarity_floor (0.05) must short-circuit even
    # though no OPENAI_API_KEY is set in the test environment (so it would
    # otherwise fall through to the extractive path).
    chunks = [_chunk(0.01)]
    answer, mode = generate_answer("unrelated question", chunks)
    assert mode == "not_found"
    assert answer == NOT_FOUND_MESSAGE


def test_extractive_fallback_below_threshold_says_not_found():
    chunks = [_chunk(0.10)]  # above the 0.05 floor, below the 0.30 threshold
    answer = generate_extractive_fallback("question", chunks)
    assert answer == NOT_FOUND_MESSAGE


def test_extractive_fallback_above_threshold_returns_passage():
    chunks = [_chunk(0.55, text="The vault opens at 9am.")]
    answer = generate_extractive_fallback("when does the vault open", chunks)
    assert "vault opens at 9am" in answer
    assert answer != NOT_FOUND_MESSAGE


def test_generate_answer_above_threshold_uses_extractive_mode():
    chunks = [_chunk(0.55, text="The vault opens at 9am.")]
    answer, mode = generate_answer("when does the vault open", chunks)
    assert mode == "extractive_fallback"
    assert "vault opens at 9am" in answer
