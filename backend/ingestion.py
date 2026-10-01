"""PDF → text → chunks → vector index.

Every function here is synchronous and CPU-bound; the API runs them in a worker thread.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from backend.vectorstore import VectorIndex

logger = logging.getLogger(__name__)

_HORIZONTAL_WHITESPACE = re.compile(r"[ \t\f\v ]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


class IngestionError(Exception):
    """Raised when a PDF cannot be turned into a searchable index. The message is user-facing."""


def clean_text(text: str) -> str:
    """Normalise whitespace while keeping paragraph breaks, which the splitter relies on."""
    lines = (_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in text.splitlines())
    return _EXCESS_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def extract_pages(path: Path) -> tuple[list[Document], int]:
    """Return one Document per page that contains text, plus the total page count.

    Page numbers in metadata are 1-based so they match what a reader sees in a PDF viewer.
    """
    try:
        reader = PdfReader(path)
    except (PdfReadError, ValueError, OSError) as exc:
        raise IngestionError("The file could not be read as a PDF. It may be corrupted.") from exc

    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty user password purely to restrict editing.
        try:
            unlocked = bool(reader.decrypt(""))
        except Exception:
            unlocked = False
        if not unlocked:
            raise IngestionError("This PDF is password-protected. Remove the password and try again.")

    try:
        raw_pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises a wide variety of errors on malformed input
        raise IngestionError("The file could not be read as a PDF. It may be corrupted.") from exc

    pages = [
        Document(page_content=text, metadata={"page": number})
        for number, text in enumerate((clean_text(t) for t in raw_pages), start=1)
        if text
    ]
    if not pages:
        raise IngestionError(
            "No selectable text was found in this PDF. Scanned or image-only documents "
            "need OCR before they can be indexed."
        )
    return pages, len(raw_pages)


def split_pages(pages: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split pages into overlapping chunks. Each chunk keeps the page number it came from."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


def build_index(
    path: Path, embeddings: Embeddings, chunk_size: int, chunk_overlap: int
) -> tuple[VectorIndex, int, int]:
    """Run the full ingestion pipeline. Returns (vector store, page count, chunk count)."""
    pages, page_count = extract_pages(path)
    chunks = split_pages(pages, chunk_size, chunk_overlap)
    logger.info("Extracted %d pages (%d with text) into %d chunks", page_count, len(pages), len(chunks))
    store = VectorIndex.from_documents(chunks, embeddings)
    return store, page_count, len(chunks)
