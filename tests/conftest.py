from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from backend.config import Settings
from backend.llm import OllamaClient
from backend.main import create_app
from tests.pdf_factory import make_pdf

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "examples" / "sample-faq.pdf"


@dataclass
class FakeOllama:
    """In-process stand-in for the Ollama HTTP API."""

    reply: list[str] = field(default_factory=lambda: ["Self-billing ", "lets you ", "bill yourself."])
    installed: list[str] = field(default_factory=lambda: ["llama3.2:3b"])
    reachable: bool = True
    chat_status: int = 200
    chat_error: str | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        if not self.reachable:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": name} for name in self.installed]})
        if request.url.path == "/api/chat":
            self.requests.append(json.loads(request.content))
            if self.chat_status != 200:
                return httpx.Response(self.chat_status, json={"error": self.chat_error or "boom"})
            lines = [{"message": {"role": "assistant", "content": part}, "done": False} for part in self.reply]
            if self.chat_error:
                lines.append({"error": self.chat_error})
            lines.append({"message": {"role": "assistant", "content": ""}, "done": True})
            return httpx.Response(200, content="\n".join(json.dumps(line) for line in lines).encode())
        return httpx.Response(404)

    def client(self) -> OllamaClient:
        return OllamaClient("http://ollama.test", "llama3.2:3b", transport=httpx.MockTransport(self.handler))


@pytest.fixture
def fake_ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", persist_documents=True, _env_file=None)


@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def make_client(
    settings: Settings, embeddings: DeterministicFakeEmbedding, fake_ollama: FakeOllama
) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(**overrides: Any) -> TestClient:
        app_settings = settings.model_copy(update=overrides) if overrides else settings
        app = create_app(app_settings, embeddings_factory=lambda: embeddings, llm=fake_ollama.client())
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


@pytest.fixture
def simple_pdf() -> bytes:
    return make_pdf(
        [
            ["Solar panels convert sunlight into electricity.", "They work best facing south."],
            [],
            ["Batteries store surplus energy for use at night."],
        ]
    )


def upload(client: TestClient, content: bytes, filename: str = "doc.pdf") -> Any:
    return client.post("/documents", files={"file": (filename, content, "application/pdf")})
