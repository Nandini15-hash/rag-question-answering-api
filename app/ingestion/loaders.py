"""Document loaders — extract raw text from an uploaded file.

Two formats are supported, per the assignment's minimum: PDF and TXT.
Each loader returns plain text; page boundaries are preserved for PDFs
(joined with a form-feed-free double newline) since we chunk on paragraph
structure downstream, not on physical pages.
"""
from pathlib import Path

from pypdf import PdfReader


class UnsupportedFileType(ValueError):
    pass


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return load_text_file(path)
    if suffix == ".pdf":
        return load_pdf_file(path)
    raise UnsupportedFileType(f"Unsupported file type: {suffix}. Supported: .txt, .pdf")
