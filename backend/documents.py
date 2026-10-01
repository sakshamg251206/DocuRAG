"""Thread-safe registry of uploaded documents, with optional on-disk persistence.

Each document lives in memory as a `DocumentRecord`. When persistence is enabled, a ready
document's vector index, chunks and metadata are written to `<data_dir>/<document id>/` so they
can be restored after a restart instead of being re-uploaded and re-embedded.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.embeddings import Embeddings

from backend.schemas import DocumentInfo, DocumentStatus
from backend.vectorstore import VectorIndex

logger = logging.getLogger(__name__)

METADATA_FILE = "document.json"


@dataclass
class DocumentRecord:
    id: str
    filename: str
    size_bytes: int
    status: DocumentStatus = DocumentStatus.PROCESSING
    pages: int | None = None
    chunks: int | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    store: VectorIndex | None = field(default=None, repr=False)

    def info(self) -> DocumentInfo:
        return DocumentInfo(
            id=self.id,
            filename=self.filename,
            status=self.status,
            size_bytes=self.size_bytes,
            pages=self.pages,
            chunks=self.chunks,
            error=self.error,
            created_at=self.created_at,
        )


class DocumentRegistry:
    def __init__(self, data_dir: Path | None = None, embedding_model: str = "") -> None:
        self._records: dict[str, DocumentRecord] = {}
        self._lock = threading.RLock()
        self._data_dir = data_dir
        # Vectors from different embedding models are not comparable, so persisted indexes
        # remember which model produced them and are skipped if the model changes.
        self._embedding_model = embedding_model

    # ── Queries ──────────────────────────────────────────────────────────────
    def get(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(document_id)

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    def count(self, status: DocumentStatus | None = None) -> int:
        with self._lock:
            return sum(1 for r in self._records.values() if status is None or r.status == status)

    # ── Mutations ────────────────────────────────────────────────────────────
    def add(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.id] = record

    def remove(self, document_id: str) -> bool:
        with self._lock:
            removed = self._records.pop(document_id, None) is not None
        if removed and self._data_dir is not None:
            shutil.rmtree(self._document_dir(document_id), ignore_errors=True)
        return removed

    def mark_ready(self, document_id: str, store: VectorIndex, pages: int, chunks: int) -> bool:
        """Attach a finished index. Returns False if the document was deleted while processing."""
        with self._lock:
            record = self._records.get(document_id)
            if record is None:
                return False
            record.store, record.pages, record.chunks = store, pages, chunks
            record.status, record.error = DocumentStatus.READY, None
        self._persist(record)
        return True

    def mark_failed(self, document_id: str, error: str) -> None:
        with self._lock:
            if record := self._records.get(document_id):
                record.status, record.error = DocumentStatus.FAILED, error

    # ── Persistence ──────────────────────────────────────────────────────────
    def _document_dir(self, document_id: str) -> Path:
        assert self._data_dir is not None
        return self._data_dir / document_id

    def _persist(self, record: DocumentRecord) -> None:
        if self._data_dir is None or record.store is None:
            return
        target = self._document_dir(record.id)
        try:
            target.mkdir(parents=True, exist_ok=True)
            record.store.save(target)
            meta = record.info().model_dump(mode="json", exclude={"status", "error"})
            meta["embedding_model"] = self._embedding_model
            (target / METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except OSError:
            logger.exception("Could not persist document %s; it will stay in memory only", record.id)
            shutil.rmtree(target, ignore_errors=True)
            return
        # The document may have been deleted while it was being written to disk.
        if self.get(record.id) is None:
            shutil.rmtree(target, ignore_errors=True)

    def load_persisted(self, embeddings: Embeddings) -> int:
        """Restore every persisted document. Corrupt entries are skipped, not fatal."""
        if self._data_dir is None or not self._data_dir.is_dir():
            return 0
        loaded = 0
        for directory in sorted(p for p in self._data_dir.iterdir() if p.is_dir()):
            meta_path = directory / METADATA_FILE
            if not meta_path.is_file():
                continue
            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
                if raw.get("embedding_model") != self._embedding_model:
                    logger.warning(
                        "Skipping %s: it was indexed with embedding model %r, but %r is configured. "
                        "Upload the document again to re-index it.",
                        directory.name,
                        raw.get("embedding_model"),
                        self._embedding_model,
                    )
                    continue
                meta = DocumentInfo.model_validate({**raw, "status": DocumentStatus.READY})
                if meta.id != directory.name:
                    raise ValueError(f"metadata id {meta.id!r} does not match directory name")
                store = VectorIndex.load(directory, embeddings)
            except Exception:
                logger.exception("Skipping unreadable persisted document in %s", directory)
                continue
            self.add(
                DocumentRecord(
                    id=meta.id,
                    filename=meta.filename,
                    size_bytes=meta.size_bytes,
                    status=DocumentStatus.READY,
                    pages=meta.pages,
                    chunks=meta.chunks,
                    created_at=meta.created_at,
                    store=store,
                )
            )
            loaded += 1
        return loaded
