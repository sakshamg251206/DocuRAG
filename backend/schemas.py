"""Request and response models exposed by the HTTP API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentInfo(BaseModel):
    id: str
    filename: str
    status: DocumentStatus
    size_bytes: int
    pages: int | None = None
    chunks: int | None = None
    error: str | None = None
    created_at: datetime


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=100)


class Source(BaseModel):
    page: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    duration_ms: int


class LLMStatus(BaseModel):
    reachable: bool
    base_url: str
    model: str
    model_available: bool
    installed_models: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    llm: LLMStatus
    embedding_model: str
    documents_total: int
    documents_ready: int


class ErrorResponse(BaseModel):
    detail: str
