"""
A small, dependency-free recursive text splitter.

Why not just use LangChain's RecursiveCharacterTextSplitter: the assignment
asks us not to reach for default RAG templates without explanation, and this
splitter is ~40 lines and easy to reason about, tune, and unit test — pulling
in a heavy framework for something this small isn't justified here.

Strategy: try to split on paragraph breaks first, then sentence breaks, then
word breaks, only falling back to a hard character cut if a single "word" is
still longer than chunk_size. This keeps chunks aligned to natural language
boundaries as much as possible, which matters for embedding quality (a chunk
that ends mid-sentence embeds worse than one that ends on a clean sentence).
Adjacent chunks overlap by `overlap` characters so a fact that sits right on
a boundary is still findable from whichever side of the boundary the query
matches.
"""
from dataclasses import dataclass

SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    index: int
    text: str


def _split_on_separator(text: str, sep: str) -> list[str]:
    if sep == " ":
        return text.split(" ")
    parts = text.split(sep)
    # keep the separator attached (except the last part) so re-joined text is faithful
    return [p + sep for p in parts[:-1]] + [parts[-1]]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # hard cut — last resort for a single very long "word" (e.g. a URL)
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest_separators = separators[0], separators[1:]
    pieces = _split_on_separator(text, sep)

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        if len(buffer) + len(piece) <= chunk_size:
            buffer += piece
        else:
            if buffer.strip():
                chunks.append(buffer)
            if len(piece) > chunk_size:
                chunks.extend(_recursive_split(piece, chunk_size, rest_separators))
                buffer = ""
            else:
                buffer = piece
    if buffer.strip():
        chunks.append(buffer)
    return chunks


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Split `text` into overlapping chunks of at most `chunk_size` characters."""
    text = text.strip()
    if not text:
        return []

    raw_chunks = _recursive_split(text, chunk_size, SEPARATORS)

    if overlap <= 0 or len(raw_chunks) <= 1:
        return [Chunk(index=i, text=c.strip()) for i, c in enumerate(raw_chunks) if c.strip()]

    overlapped: list[str] = []
    for i, c in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(c)
            continue
        prev_tail = raw_chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + c)

    return [Chunk(index=i, text=c.strip()) for i, c in enumerate(overlapped) if c.strip()]
