"""Sentence-transformer embeddings, loaded once per process.

Loading the model takes several seconds, so it is cached and shared by every document
instead of being re-created for each upload.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)
_load_lock = threading.Lock()


def load_embeddings(model_name: str) -> Embeddings:
    # Several uploads can start at once; the lock ensures the model is only loaded once.
    with _load_lock:
        return _load(model_name)


@lru_cache(maxsize=4)
def _load(model_name: str) -> Embeddings:
    # Imported lazily: torch/sentence-transformers are heavy and not needed by the test suite.
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("Loading embedding model '%s'", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        # Unit-length vectors make the index's inner-product score a cosine similarity.
        encode_kwargs={"normalize_embeddings": True},
    )
