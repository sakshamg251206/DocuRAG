from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from backend.rag import SYSTEM_PROMPT, build_messages, retrieve, sources_from
from backend.schemas import ChatMessage
from backend.vectorstore import VectorIndex


def doc(text: str, page: int) -> Document:
    return Document(page_content=text, metadata={"page": page})


def test_build_messages_places_context_in_final_user_turn() -> None:
    messages = build_messages("  What is X?  ", [doc("X is a letter.", 4)], [], max_history_turns=6)

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[-1]["role"] == "user"
    assert "[Excerpt 1 | page 4]\nX is a letter." in messages[-1]["content"]
    assert messages[-1]["content"].endswith("Question: What is X?")


def test_build_messages_keeps_only_recent_history() -> None:
    history = [ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"turn {i}") for i in range(10)]

    messages = build_messages("next?", [], history, max_history_turns=4)

    assert [m["content"] for m in messages[1:-1]] == ["turn 6", "turn 7", "turn 8", "turn 9"]
    assert "No relevant excerpts" in messages[-1]["content"]


def test_build_messages_can_disable_history() -> None:
    history = [ChatMessage(role="user", content="old")]
    assert len(build_messages("q", [], history, max_history_turns=0)) == 2


def test_sources_are_deduplicated_sorted_and_truncated() -> None:
    long_text = "lorem ipsum " * 100
    sources = sources_from([doc("b", 7), doc(long_text, 2), doc("again", 7)])

    assert [s.page for s in sources] == [2, 7]
    assert sources[1].excerpt == "b"
    assert len(sources[0].excerpt) <= 301 and sources[0].excerpt.endswith("…")


def test_retrieve_caps_k_to_index_size() -> None:
    store = VectorIndex.from_documents([doc("alpha", 1), doc("beta", 2)], DeterministicFakeEmbedding(size=8))

    results = retrieve(store, "alpha", k=5, fetch_k=20)

    assert len(results) == 2
