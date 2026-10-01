"""Minimal async client for the Ollama HTTP API."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from backend.schemas import LLMStatus

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the language model cannot produce an answer. The message is user-facing."""


def _model_matches(configured: str, installed: str) -> bool:
    """Ollama reports untagged models as `name:latest`; treat `name` and `name:latest` as equal."""
    if configured == installed:
        return True
    return ":" not in configured and installed == f"{configured}:latest"


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        timeout: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        # Generous read timeout for slow CPUs, but fail fast if Ollama isn't listening at all.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def status(self) -> LLMStatus:
        installed: list[str] = []
        reachable = False
        try:
            response = await self._client.get("/api/tags", timeout=3.0)
            response.raise_for_status()
            installed = sorted(m["name"] for m in response.json().get("models", []) if "name" in m)
            reachable = True
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.debug("Ollama status check failed: %s", exc)
        return LLMStatus(
            reachable=reachable,
            base_url=self.base_url,
            model=self.model,
            model_available=any(_model_matches(self.model, name) for name in installed),
            installed_models=installed,
        )

    async def stream_chat(self, messages: Sequence[dict[str, str]]) -> AsyncIterator[str]:
        """Yield response text fragments as Ollama generates them."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise LLMError(self._describe_http_error(response))
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed line from Ollama: %r", line[:200])
                        continue
                    if error := data.get("error"):
                        raise LLMError(f"The language model reported an error: {error}")
                    if fragment := data.get("message", {}).get("content", ""):
                        yield fragment
                    if data.get("done"):
                        return
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.base_url}. Make sure it is running (`ollama serve`)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError("The language model took too long to respond. Please try again.") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Lost connection to the language model: {exc}") from exc

    async def chat(self, messages: Sequence[dict[str, str]]) -> str:
        return "".join([fragment async for fragment in self.stream_chat(messages)])

    def _describe_http_error(self, response: httpx.Response) -> str:
        try:
            detail = str(response.json().get("error", ""))
        except ValueError:
            detail = response.text
        if response.status_code == 404 or "not found" in detail.lower():
            return f"The model '{self.model}' is not installed in Ollama. Run `ollama pull {self.model}`."
        return f"The language model returned HTTP {response.status_code}: {detail or 'unknown error'}"
