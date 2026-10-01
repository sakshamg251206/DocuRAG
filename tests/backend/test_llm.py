from __future__ import annotations

import re

import pytest

from backend.llm import LLMError, _model_matches
from tests.conftest import FakeOllama

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    ("configured", "installed", "expected"),
    [
        ("llama3.2:3b", "llama3.2:3b", True),
        ("llama3.2", "llama3.2:latest", True),
        ("llama3.2:3b", "llama3.2:latest", False),
        ("mistral", "llama3.2:latest", False),
    ],
)
def test_model_matching(configured: str, installed: str, expected: bool) -> None:
    assert _model_matches(configured, installed) is expected


async def test_status_reports_installed_models(fake_ollama: FakeOllama) -> None:
    fake_ollama.installed = ["mistral:latest", "llama3.2:3b"]
    status = await fake_ollama.client().status()

    assert status.reachable and status.model_available
    assert status.installed_models == ["llama3.2:3b", "mistral:latest"]


async def test_status_when_unreachable(fake_ollama: FakeOllama) -> None:
    fake_ollama.reachable = False
    status = await fake_ollama.client().status()

    assert not status.reachable and not status.model_available


async def test_chat_streams_and_sends_options(fake_ollama: FakeOllama) -> None:
    client = fake_ollama.client()
    fragments = [f async for f in client.stream_chat(MESSAGES)]

    assert fragments == fake_ollama.reply
    sent = fake_ollama.requests[0]
    assert sent["stream"] is True and sent["model"] == "llama3.2:3b"
    assert sent["options"] == {"temperature": 0.1}


async def test_missing_model_gives_actionable_error(fake_ollama: FakeOllama) -> None:
    fake_ollama.chat_status = 404
    fake_ollama.chat_error = "model 'llama3.2:3b' not found"

    with pytest.raises(LLMError, match=re.escape("ollama pull llama3.2:3b")):
        await fake_ollama.client().chat(MESSAGES)


async def test_connection_error_is_wrapped(fake_ollama: FakeOllama) -> None:
    fake_ollama.reachable = False

    with pytest.raises(LLMError, match="Cannot reach Ollama"):
        await fake_ollama.client().chat(MESSAGES)


async def test_error_mid_stream_is_raised(fake_ollama: FakeOllama) -> None:
    fake_ollama.chat_error = "out of memory"

    with pytest.raises(LLMError, match="out of memory"):
        await fake_ollama.client().chat(MESSAGES)
