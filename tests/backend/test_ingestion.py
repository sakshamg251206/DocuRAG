from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from backend.ingestion import IngestionError, build_index, clean_text, extract_pages, split_pages
from tests.conftest import SAMPLE_PDF
from tests.pdf_factory import make_pdf


def write(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "doc.pdf"
    path.write_bytes(content)
    return path


def test_clean_text_collapses_spaces_but_keeps_paragraphs() -> None:
    assert clean_text("  a \t b  \n\n\n\n c d ") == "a b\n\nc d"


def test_extract_pages_uses_one_based_numbers_and_skips_blank_pages(tmp_path: Path) -> None:
    path = write(tmp_path, make_pdf([["first page"], [], ["third page"]]))

    pages, total = extract_pages(path)

    assert total == 3
    assert [p.metadata["page"] for p in pages] == [1, 3]
    assert pages[0].page_content == "first page"


def test_extract_pages_rejects_pdf_without_text(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="No selectable text"):
        extract_pages(write(tmp_path, make_pdf([[], []])))


def test_extract_pages_rejects_corrupt_file(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="could not be read"):
        extract_pages(write(tmp_path, b"%PDF-1.4\nthis is not really a pdf"))


def test_split_pages_preserves_page_metadata(tmp_path: Path) -> None:
    long_line = "word " * 60
    pages, _ = extract_pages(write(tmp_path, make_pdf([[long_line] * 10, ["short"]])))

    chunks = split_pages(pages, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 2
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert {c.metadata["page"] for c in chunks} == {1, 2}


def test_build_index_on_sample_document() -> None:
    store, pages, chunks = build_index(SAMPLE_PDF, DeterministicFakeEmbedding(size=16), 1000, 200)

    assert pages == 16
    assert chunks == len(store) > pages
