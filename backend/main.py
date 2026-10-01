"""HTTP API: upload PDFs, track their indexing, and ask questions about them."""

from __future__ import annotations

import json
import logging
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path, PurePath
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend import __version__
from backend.config import Settings, get_settings
from backend.documents import DocumentRecord, DocumentRegistry
from backend.embeddings import load_embeddings
from backend.ingestion import IngestionError, build_index
from backend.llm import LLMError, OllamaClient
from backend.rag import build_messages, retrieve, sources_from
from backend.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    DocumentStatus,
    ErrorResponse,
    HealthResponse,
)

logger = logging.getLogger("docurag")

PDF_MAGIC = b"%PDF-"
UPLOAD_READ_CHUNK = 1024 * 1024


class AppState:
    """Long-lived services shared by all requests."""

    def __init__(self, settings: Settings, embeddings_factory: Callable[[], Embeddings], llm: OllamaClient) -> None:
        self.settings = settings
        self.embeddings_factory = embeddings_factory
        self.llm = llm
        self.registry = DocumentRegistry(
            settings.data_dir if settings.persist_documents else None, embedding_model=settings.embed_model
        )


def create_app(
    settings: Settings | None = None,
    embeddings_factory: Callable[[], Embeddings] | None = None,
    llm: OllamaClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    state = AppState(
        settings=settings,
        embeddings_factory=embeddings_factory or (lambda: load_embeddings(settings.embed_model)),
        llm=llm
        or OllamaClient(
            settings.ollama_base_url,
            settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        ),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await _restore_documents(state)
        llm_status = await state.llm.status()
        if not llm_status.reachable:
            logger.warning("Ollama is not reachable at %s. Start it with `ollama serve`.", llm_status.base_url)
        elif not llm_status.model_available:
            logger.warning("Model '%s' is not installed. Run `ollama pull %s`.", llm_status.model, llm_status.model)
        else:
            logger.info("Ollama is ready with model '%s'", llm_status.model)
        yield
        await state.llm.aclose()

    app = FastAPI(
        title="DocuRAG API",
        version=__version__,
        description="Chat with PDF documents using retrieval-augmented generation and a local LLM.",
        lifespan=lifespan,
        responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
    )
    app.state.docurag = state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    _register_routes(app)
    return app


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.docurag
    return state


StateDep = Annotated[AppState, Depends(get_state)]


async def _restore_documents(state: AppState) -> None:
    if not state.settings.persist_documents:
        return
    data_dir = state.settings.data_dir
    if not data_dir.is_dir() or not any(data_dir.iterdir()):
        return
    try:
        embeddings = await run_in_threadpool(state.embeddings_factory)
        restored = await run_in_threadpool(state.registry.load_persisted, embeddings)
        logger.info("Restored %d document(s) from %s", restored, data_dir)
    except Exception:
        logger.exception("Could not restore persisted documents; continuing with an empty library")


def _ingest(state: AppState, document_id: str, path: Path) -> None:
    """Background job: build the vector index for an uploaded PDF."""
    started = time.perf_counter()
    try:
        try:
            embeddings = state.embeddings_factory()
        except Exception as exc:
            raise IngestionError(
                f"The embedding model '{state.settings.embed_model}' could not be loaded. "
                "Check the server logs and your internet connection (it is downloaded on first use)."
            ) from exc
        store, pages, chunks = build_index(
            path,
            embeddings,
            chunk_size=state.settings.chunk_size,
            chunk_overlap=state.settings.chunk_overlap,
        )
        if state.registry.mark_ready(document_id, store, pages, chunks):
            logger.info("Document %s ready in %.1fs", document_id, time.perf_counter() - started)
    except IngestionError as exc:
        logger.warning("Document %s could not be indexed: %s", document_id, exc, exc_info=exc.__cause__)
        state.registry.mark_failed(document_id, str(exc))
    except Exception:
        logger.exception("Unexpected error while indexing document %s", document_id)
        state.registry.mark_failed(document_id, "An unexpected error occurred while indexing this document.")
    finally:
        path.unlink(missing_ok=True)


def _require_ready(state: AppState, document_id: str) -> DocumentRecord:
    record = state.registry.get(document_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    if record.status != DocumentStatus.READY or record.store is None:
        detail = (
            "This document is still being indexed. Please wait a moment."
            if record.status == DocumentStatus.PROCESSING
            else f"This document could not be indexed: {record.error}"
        )
        raise HTTPException(status.HTTP_409_CONFLICT, detail)
    return record


async def _prepare(
    state: AppState, document_id: str, request: AskRequest
) -> tuple[list[Document], list[dict[str, str]]]:
    record = _require_ready(state, document_id)
    assert record.store is not None
    # Embedding the question and searching FAISS is CPU-bound; keep it off the event loop.
    docs = await run_in_threadpool(
        retrieve, record.store, request.question, state.settings.retrieval_k, state.settings.retrieval_fetch_k
    )
    messages = build_messages(request.question, docs, request.history, state.settings.history_turns)
    return docs, messages


def _sse(event: str, data: object) -> str:
    """Encode one Server-Sent Event. Data is JSON so newlines in tokens can't break framing."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _safe_filename(raw: str | None) -> str:
    name = PurePath((raw or "").replace("\\", "/")).name.strip()
    return name[:255] or "document.pdf"


def _register_routes(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"name": "DocuRAG API", "version": __version__, "docs": "/docs"}

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(state: StateDep) -> HealthResponse:
        return HealthResponse(
            version=__version__,
            llm=await state.llm.status(),
            embedding_model=state.settings.embed_model,
            documents_total=state.registry.count(),
            documents_ready=state.registry.count(DocumentStatus.READY),
        )

    @app.get("/documents", response_model=list[DocumentInfo], tags=["documents"])
    async def list_documents(state: StateDep) -> list[DocumentInfo]:
        return [record.info() for record in state.registry.list()]

    @app.post(
        "/documents",
        response_model=DocumentInfo,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["documents"],
        responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    )
    async def upload_document(
        state: StateDep,
        background_tasks: BackgroundTasks,
        file: Annotated[UploadFile, File(description="A PDF file with selectable text.")],
    ) -> DocumentInfo:
        """Upload a PDF. Indexing happens in the background; poll `GET /documents/{id}` for status."""
        filename = _safe_filename(file.filename)
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only PDF files are supported.")

        limit = state.settings.max_pdf_size_bytes
        size = 0
        with tempfile.NamedTemporaryFile(prefix="docurag-", suffix=".pdf", delete=False) as tmp:
            path = Path(tmp.name)
            try:
                while chunk := await file.read(UPLOAD_READ_CHUNK):
                    if size == 0 and not chunk.startswith(PDF_MAGIC):
                        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This file is not a valid PDF.")
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(
                            status.HTTP_413_CONTENT_TOO_LARGE,
                            f"PDF exceeds the {state.settings.max_pdf_size_mb} MB size limit.",
                        )
                    tmp.write(chunk)
                if size == 0:
                    raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
            except BaseException:
                tmp.close()
                path.unlink(missing_ok=True)
                raise

        record = DocumentRecord(id=str(uuid.uuid4()), filename=filename, size_bytes=size)
        state.registry.add(record)
        background_tasks.add_task(_ingest, state, record.id, path)
        logger.info("Accepted '%s' (%.1f MB) as %s", filename, size / 1024 / 1024, record.id)
        return record.info()

    @app.get("/documents/{document_id}", response_model=DocumentInfo, tags=["documents"])
    async def get_document(document_id: str, state: StateDep) -> DocumentInfo:
        record = state.registry.get(document_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
        return record.info()

    @app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"])
    async def delete_document(document_id: str, state: StateDep) -> None:
        if not state.registry.remove(document_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
        logger.info("Deleted document %s", document_id)

    @app.post(
        "/documents/{document_id}/ask",
        response_model=AskResponse,
        tags=["chat"],
        responses={409: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    )
    async def ask(document_id: str, request: AskRequest, state: StateDep) -> AskResponse:
        """Ask a question and receive the complete answer in one response."""
        started = time.perf_counter()
        docs, messages = await _prepare(state, document_id, request)
        try:
            answer = await state.llm.chat(messages)
        except LLMError as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
        return AskResponse(
            answer=answer.strip(),
            sources=sources_from(docs),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    @app.post(
        "/documents/{document_id}/ask/stream",
        tags=["chat"],
        response_class=StreamingResponse,
        responses={
            200: {
                "content": {"text/event-stream": {}},
                "description": "Server-Sent Events: `sources`, then `token`s, then `done` (or `error`).",
            },
            409: {"model": ErrorResponse},
        },
    )
    async def ask_stream(document_id: str, request: AskRequest, state: StateDep) -> StreamingResponse:
        """Ask a question and stream the answer token by token as Server-Sent Events."""
        started = time.perf_counter()
        docs, messages = await _prepare(state, document_id, request)
        sources = [source.model_dump() for source in sources_from(docs)]

        async def events() -> AsyncIterator[str]:
            yield _sse("sources", sources)
            try:
                async for fragment in state.llm.stream_chat(messages):
                    yield _sse("token", fragment)
            except LLMError as exc:
                yield _sse("error", {"detail": str(exc)})
                return
            except Exception:
                logger.exception("Unexpected error while streaming an answer")
                yield _sse("error", {"detail": "An unexpected error occurred while generating the answer."})
                return
            yield _sse("done", {"duration_ms": int((time.perf_counter() - started) * 1000)})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


app = create_app()
