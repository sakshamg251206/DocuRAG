"""UI tests that run the real Streamlit script against a stubbed API client."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from api_client import ApiError, StreamEvent

APP = str(Path(__file__).resolve().parents[2] / "frontend" / "app.py")

HEALTHY: dict[str, Any] = {
    "status": "ok",
    "version": "3.0.0",
    "embedding_model": "all-MiniLM-L6-v2",
    "llm": {
        "reachable": True,
        "model_available": True,
        "model": "llama3.2:3b",
        "base_url": "http://localhost:11434",
        "installed_models": ["llama3.2:3b"],
    },
}
READY_DOC: dict[str, Any] = {
    "id": "doc-1",
    "filename": "handbook.pdf",
    "status": "ready",
    "size_bytes": 2048,
    "pages": 12,
    "chunks": 40,
    "error": None,
    "created_at": "2026-01-01T00:00:00Z",
}


class StubClient:
    def __init__(self, health: dict[str, Any] | None = None, documents: list[dict[str, Any]] | None = None) -> None:
        self._health = health
        self.documents = documents or []
        self.questions: list[tuple[str, list[dict[str, str]]]] = []

    def health(self) -> dict[str, Any]:
        if self._health is None:
            raise ApiError("Cannot connect to the DocuRAG API.")
        return self._health

    def list_documents(self) -> list[dict[str, Any]]:
        return self.documents

    def get_document(self, document_id: str) -> dict[str, Any]:
        return next(d for d in self.documents if d["id"] == document_id)

    def ask_stream(self, document_id: str, question: str, history: list[dict[str, str]]) -> Iterator[StreamEvent]:
        self.questions.append((question, history))
        yield StreamEvent("sources", [{"page": 3, "excerpt": "Holidays are listed on page three."}])
        yield StreamEvent("token", "There are ")
        yield StreamEvent("token", "12 holidays.")
        yield StreamEvent("done", {"duration_ms": 10})


def run(stub: StubClient, **state: Any) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=15)
    for key, value in state.items():
        at.session_state[key] = value
    with patch("api_client.ApiClient", return_value=stub):
        at.run()
    return at


@pytest.fixture(autouse=True)
def _clear_streamlit_caches() -> Iterator[None]:
    import streamlit as st

    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def all_text(at: AppTest) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.title]
    parts += [e.value for e in at.error] + [e.value for e in at.warning] + [e.value for e in at.subheader]
    return "\n".join(str(p) for p in parts)


def test_welcome_screen_explains_the_product() -> None:
    at = run(StubClient(HEALTHY))

    assert not at.exception
    assert at.title[0].value == "Chat with your PDFs"
    text = all_text(at)
    assert "1. Upload a PDF" in text and "3. Ask questions" in text


def test_offline_api_shows_how_to_start_it() -> None:
    at = run(StubClient(health=None))

    assert not at.exception
    assert "Can't reach the DocuRAG API" in at.error[0].value


def test_missing_model_shows_pull_command() -> None:
    health = {**HEALTHY, "llm": {**HEALTHY["llm"], "model_available": False}}

    at = run(StubClient(health, [READY_DOC]), active_id="doc-1")

    assert "ollama pull llama3.2:3b" in at.warning[0].value
    assert at.chat_input[0].disabled


def test_ready_document_offers_starter_questions() -> None:
    at = run(StubClient(HEALTHY, [READY_DOC]), active_id="doc-1")

    assert not at.exception
    assert "handbook" in at.subheader[0].value
    assert any("Summarize" in b.label for b in at.button)
    assert not at.chat_input[0].disabled


def test_asking_a_question_streams_answer_with_sources() -> None:
    stub = StubClient(HEALTHY, [READY_DOC])
    at = run(stub, active_id="doc-1")

    at.chat_input[0].set_value("How many holidays?")
    with patch("api_client.ApiClient", return_value=stub):
        at.run()

    assert not at.exception
    assert stub.questions == [("How many holidays?", [])]
    messages = at.session_state["chats"]["doc-1"]
    assert messages[0] == {"role": "user", "content": "How many holidays?"}
    assert messages[1]["content"] == "There are 12 holidays."
    assert messages[1]["sources"][0]["page"] == 3
    assert "Sources · page 3" in repr(at.main)  # expanders with icons are exposed as `Status` blocks


def test_failed_document_shows_reason() -> None:
    failed = {**READY_DOC, "status": "failed", "error": "No selectable text was found."}

    at = run(StubClient(HEALTHY, [failed]), active_id="doc-1")

    assert "No selectable text" in at.error[0].value


def test_starter_question_is_asked_and_suggestions_disappear() -> None:
    stub = StubClient(HEALTHY, [READY_DOC])
    at = run(stub, active_id="doc-1")

    with patch("api_client.ApiClient", return_value=stub):
        at.button(key="starter_0").click().run()

    assert stub.questions[0][0].startswith("Summarize")
    assert not any(b.key == "starter_0" for b in at.button)


def test_follow_up_questions_send_history() -> None:
    stub = StubClient(HEALTHY, [READY_DOC])
    at = run(stub, active_id="doc-1")
    for question in ("First?", "Second?"):
        at.chat_input[0].set_value(question)
        with patch("api_client.ApiClient", return_value=stub):
            at.run()

    assert stub.questions[1][1] == [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "There are 12 holidays."},
    ]
