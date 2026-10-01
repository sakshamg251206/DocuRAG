"""Runtime configuration, loaded from environment variables (and an optional `.env` file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables for the API. Every field can be overridden with an env var of the same name."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Language model (served by Ollama) ────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=300.0, gt=0)

    # ── Embeddings & retrieval ───────────────────────────────────────────────
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    retrieval_k: int = Field(default=5, ge=1, le=50)
    retrieval_fetch_k: int = Field(default=20, ge=1, le=200)
    history_turns: int = Field(default=6, ge=0, le=50)

    # ── Uploads & storage ────────────────────────────────────────────────────
    max_pdf_size_mb: int = Field(default=50, ge=1)
    data_dir: Path = Path("data")
    persist_documents: bool = True

    # ── HTTP ─────────────────────────────────────────────────────────────────
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:8501"]
    log_level: str = "INFO"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string (the natural format for an env var)."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.retrieval_fetch_k < self.retrieval_k:
            raise ValueError("RETRIEVAL_FETCH_K must be greater than or equal to RETRIEVAL_K")
        return self

    @property
    def max_pdf_size_bytes(self) -> int:
        return self.max_pdf_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
