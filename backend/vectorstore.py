"""A small FAISS-backed vector index for document chunks.

Vectors are L2-normalised and stored in an exact inner-product index, so search scores are
cosine similarities. Chunks are persisted as plain JSON next to the FAISS index, which keeps the
on-disk format transparent and avoids unpickling anything when documents are restored.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.utils import maximal_marginal_relevance

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


def _as_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2:
        raise ValueError("expected a 2-D array of embeddings")
    faiss.normalize_L2(matrix)
    return matrix


class VectorIndex:
    def __init__(self, index: faiss.Index, documents: list[Document], embeddings: Embeddings) -> None:
        if index.ntotal != len(documents):
            raise ValueError(f"index has {index.ntotal} vectors but {len(documents)} documents were given")
        self._index = index
        self._documents = documents
        self._embeddings = embeddings

    @classmethod
    def from_documents(cls, documents: Sequence[Document], embeddings: Embeddings) -> VectorIndex:
        if not documents:
            raise ValueError("cannot build an index without documents")
        matrix = _as_matrix(embeddings.embed_documents([d.page_content for d in documents]))
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index, list(documents), embeddings)

    def __len__(self) -> int:
        return len(self._documents)

    def search(self, query: str, k: int, fetch_k: int, diversity: float = 0.5) -> list[Document]:
        """Maximal Marginal Relevance search.

        Fetches the `fetch_k` most similar chunks, then greedily selects `k` of them, trading off
        similarity to the query against similarity to chunks already chosen (`diversity` 0 → pure
        diversity, 1 → pure relevance). This avoids filling the prompt with near-duplicates.
        """
        k = min(k, len(self))
        if k <= 0:
            return []
        fetch_k = min(max(fetch_k, k), len(self))
        query_vector = _as_matrix([self._embeddings.embed_query(query)])
        _, ids = self._index.search(query_vector, fetch_k)
        candidates = [int(i) for i in ids[0] if i >= 0]
        candidate_vectors = np.vstack([self._index.reconstruct(i) for i in candidates])
        chosen = maximal_marginal_relevance(query_vector[0], candidate_vectors.tolist(), lambda_mult=diversity, k=k)
        return [self._documents[candidates[i]] for i in chosen]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / INDEX_FILE))
        chunks = [{"text": d.page_content, "metadata": d.metadata} for d in self._documents]
        (directory / CHUNKS_FILE).write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embeddings: Embeddings) -> VectorIndex:
        index = faiss.read_index(str(directory / INDEX_FILE))
        chunks = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
        documents = [Document(page_content=c["text"], metadata=c.get("metadata", {})) for c in chunks]
        return cls(index, documents, embeddings)
