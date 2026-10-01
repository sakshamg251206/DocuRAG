from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend.vectorstore import CHUNKS_FILE, VectorIndex


class KeywordEmbeddings(Embeddings):
    """Embeds text as keyword counts, so similarity is predictable in tests."""

    vocabulary = ("solar", "battery", "wind", "grid")

    def _embed(self, text: str) -> list[float]:
        words = text.lower().split()
        return [float(words.count(term)) + 0.01 for term in self.vocabulary]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


DOCS = [
    Document(page_content="solar solar panels", metadata={"page": 1}),
    Document(page_content="solar solar panels again", metadata={"page": 2}),
    Document(page_content="battery storage", metadata={"page": 3}),
    Document(page_content="wind turbines", metadata={"page": 4}),
]


def test_most_relevant_chunk_comes_first() -> None:
    index = VectorIndex.from_documents(DOCS, KeywordEmbeddings())

    assert index.search("battery", k=1, fetch_k=4)[0].metadata["page"] == 3


def test_mmr_prefers_diverse_results_over_duplicates() -> None:
    index = VectorIndex.from_documents(DOCS, KeywordEmbeddings())

    pages = [d.metadata["page"] for d in index.search("solar battery", k=2, fetch_k=4, diversity=0.3)]

    assert 3 in pages, "the second result should not be a near-duplicate solar chunk"


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    index = VectorIndex.from_documents(DOCS, KeywordEmbeddings())
    index.save(tmp_path)

    restored = VectorIndex.load(tmp_path, KeywordEmbeddings())

    assert len(restored) == len(DOCS)
    assert restored.search("wind", k=1, fetch_k=4)[0].metadata == {"page": 4}
    assert "solar solar panels" in (tmp_path / CHUNKS_FILE).read_text()


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="without documents"):
        VectorIndex.from_documents([], KeywordEmbeddings())
