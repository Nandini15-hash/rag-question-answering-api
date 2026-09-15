from app.ingestion.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=100, overlap=10) == []


def test_short_text_returns_single_chunk():
    chunks = chunk_text("hello world", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_long_text_splits_into_multiple_chunks_within_size():
    text = ("Paragraph one. " * 20) + "\n\n" + ("Paragraph two. " * 20)
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        # allow a little slack: overlap can push slightly over chunk_size
        assert len(c.text) <= 200 + 20 + 5


def test_overlap_creates_shared_text_between_adjacent_chunks():
    text = "A" * 50 + " " + "B" * 50 + " " + "C" * 50 + " " + "D" * 50
    chunks = chunk_text(text, chunk_size=60, overlap=15)
    assert len(chunks) >= 2
    # the tail of chunk i should reappear at the head of chunk i+1
    for i in range(len(chunks) - 1):
        tail = chunks[i].text[-10:]
        assert tail in chunks[i + 1].text or chunks[i + 1].text.startswith(tail[:5])


def test_chunk_indices_are_sequential():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert [c.index for c in chunks] == list(range(len(chunks)))
