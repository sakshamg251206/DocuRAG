from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from api_client import ApiClient, ApiError, StreamEvent, parse_sse


def response(status: int = 200, json_body: Any = None, lines: list[str] | None = None) -> MagicMock:
    mock = MagicMock(spec=requests.Response)
    mock.status_code = status
    mock.ok = status < 400
    mock.json.return_value = json_body
    mock.iter_lines.return_value = iter(lines or [])
    return mock


def client_with(result: Any) -> tuple[ApiClient, MagicMock]:
    session = MagicMock(spec=requests.Session)
    if isinstance(result, Exception):
        session.request.side_effect = result
    else:
        session.request.return_value = result
    return ApiClient("http://api.test/", session=session), session


def test_parse_sse_handles_events_comments_and_multiline_data() -> None:
    lines = [
        ": keep-alive",
        "event: sources",
        'data: [{"page": 1}]',
        "",
        "event: token",
        'data: "a\\nb"',
        "",
        "data: 1",
    ]

    assert list(parse_sse(lines)) == [
        StreamEvent("sources", [{"page": 1}]),
        StreamEvent("token", "a\nb"),
        StreamEvent("message", 1),
    ]


def test_successful_request_returns_json() -> None:
    client, session = client_with(response(json_body=[{"id": "1"}]))

    assert client.list_documents() == [{"id": "1"}]
    method, url = session.request.call_args.args
    assert (method, url) == ("GET", "http://api.test/documents")


def test_api_error_uses_detail_message() -> None:
    client, _ = client_with(response(409, {"detail": "Still indexing."}))

    with pytest.raises(ApiError, match="Still indexing") as info:
        client.get_document("x")
    assert info.value.status_code == 409


def test_validation_errors_get_friendly_message() -> None:
    client, _ = client_with(response(422, {"detail": [{"loc": ["body"], "msg": "bad"}]}))

    with pytest.raises(ApiError, match="not valid"):
        client.upload("a.pdf", b"%PDF-")


def test_connection_failure_is_reported_clearly() -> None:
    client, _ = client_with(requests.ConnectionError("refused"))

    with pytest.raises(ApiError, match=re.escape("Cannot connect to the DocuRAG API at http://api.test")):
        client.health()


def test_ask_stream_yields_parsed_events_and_closes_response() -> None:
    stream = response(lines=["event: token", 'data: "Hi"', "", "event: done", 'data: {"duration_ms": 5}', ""])
    client, session = client_with(stream)

    events = list(client.ask_stream("doc", "q?", [{"role": "user", "content": "earlier"}]))

    assert [e.event for e in events] == ["token", "done"]
    assert session.request.call_args.kwargs["json"] == {
        "question": "q?",
        "history": [{"role": "user", "content": "earlier"}],
    }
    stream.close.assert_called_once()
