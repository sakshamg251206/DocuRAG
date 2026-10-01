"""Typed HTTP client for the DocuRAG API, kept free of Streamlit so it can be unit-tested."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import requests


class ApiError(Exception):
    """A user-presentable failure talking to the API."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class StreamEvent:
    event: str
    data: Any


def parse_sse(lines: Iterable[str]) -> Iterator[StreamEvent]:
    """Parse a Server-Sent Events stream whose `data` fields are JSON."""
    event, data_lines = "message", list[str]()
    for line in lines:
        if line == "":
            if data_lines:
                yield StreamEvent(event, json.loads("\n".join(data_lines)))
            event, data_lines = "message", list[str]()
        elif line.startswith(":"):
            continue  # comment / keep-alive
        else:
            name, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if name == "event":
                event = value
            elif name == "data":
                data_lines.append(value)
    if data_lines:
        yield StreamEvent(event, json.loads("\n".join(data_lines)))


def _error_message(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str) and detail:
        return detail
    if response.status_code == 422:
        return "The request was not valid. Please check your input."
    return f"The server responded with an error (HTTP {response.status_code})."


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 15.0, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()

    def _request(
        self, method: str, path: str, *, timeout: float | tuple[float, float] | None = None, **kwargs: Any
    ) -> requests.Response:
        try:
            response = self._session.request(
                method, f"{self.base_url}{path}", timeout=timeout or self.timeout, **kwargs
            )
        except requests.ConnectionError as exc:
            raise ApiError(f"Cannot connect to the DocuRAG API at {self.base_url}.") from exc
        except requests.Timeout as exc:
            raise ApiError("The DocuRAG API did not respond in time.") from exc
        except requests.RequestException as exc:
            raise ApiError(f"Request to the DocuRAG API failed: {exc}") from exc
        if not response.ok:
            message = _error_message(response)
            response.close()
            raise ApiError(message, response.status_code)
        return response

    def health(self) -> dict[str, Any]:
        result: dict[str, Any] = self._request("GET", "/health", timeout=5).json()
        return result

    def list_documents(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._request("GET", "/documents").json()
        return result

    def get_document(self, document_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._request("GET", f"/documents/{document_id}").json()
        return result

    def upload(self, filename: str, content: bytes) -> dict[str, Any]:
        files = {"file": (filename, content, "application/pdf")}
        result: dict[str, Any] = self._request("POST", "/documents", files=files, timeout=120).json()
        return result

    def delete(self, document_id: str) -> None:
        self._request("DELETE", f"/documents/{document_id}")

    def ask_stream(
        self, document_id: str, question: str, history: list[dict[str, str]], timeout: float = 300
    ) -> Iterator[StreamEvent]:
        """Stream answer events. The connection timeout is short; the read timeout allows slow CPUs."""
        response = self._request(
            "POST",
            f"/documents/{document_id}/ask/stream",
            json={"question": question, "history": history},
            stream=True,
            timeout=(5, timeout),
        )
        try:
            lines = response.iter_lines(decode_unicode=True)
            yield from parse_sse(line.decode() if isinstance(line, bytes) else (line or "") for line in lines)
        except requests.RequestException as exc:
            raise ApiError("The connection was interrupted while the answer was being generated.") from exc
        finally:
            response.close()
