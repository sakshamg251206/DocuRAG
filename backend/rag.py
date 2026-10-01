"""Retrieval and prompt construction: the "R" and "A" of RAG."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document

from backend.schemas import ChatMessage, Source
from backend.vectorstore import VectorIndex

SYSTEM_PROMPT = """You are DocuRAG, an assistant that answers questions about a single document.

Rules:
- Answer using ONLY the information in the provided document excerpts.
- If the excerpts do not contain the answer, say clearly that the document does not cover it. Never invent facts.
- When it helps the reader, mention the page number(s) your answer comes from, e.g. "(page 3)".
- Be concise and well structured. Use short paragraphs, and "- " bullet points for lists or steps.
- Earlier conversation turns are provided only to resolve follow-up questions such as "what about the second one?"."""

EXCERPT_LENGTH = 300


def retrieve(store: VectorIndex, question: str, k: int, fetch_k: int) -> list[Document]:
    """Find the `k` chunks most relevant to the question, using MMR for diversity."""
    return store.search(question, k=k, fetch_k=fetch_k)


def format_context(docs: Sequence[Document]) -> str:
    return "\n\n".join(
        f"[Excerpt {i} | page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def build_messages(
    question: str,
    docs: Sequence[Document],
    history: Sequence[ChatMessage],
    max_history_turns: int,
) -> list[dict[str, str]]:
    """Assemble the chat messages sent to the LLM.

    The retrieved context is attached to the *current* question only, so the prompt stays small
    even in long conversations.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    recent = list(history)[-max_history_turns:] if max_history_turns > 0 else []
    messages.extend({"role": turn.role, "content": turn.content} for turn in recent)

    context = format_context(docs) if docs else "(No relevant excerpts were found in the document.)"
    messages.append(
        {
            "role": "user",
            "content": f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {question.strip()}",
        }
    )
    return messages


def _excerpt(text: str, limit: int = EXCERPT_LENGTH) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def sources_from(docs: Sequence[Document]) -> list[Source]:
    """One citation per page, ordered by page number, with a short preview of the matched text."""
    by_page: dict[int, str] = {}
    for doc in docs:
        page = int(doc.metadata.get("page", 0))
        by_page.setdefault(page, doc.page_content)
    return [Source(page=page, excerpt=_excerpt(text)) for page, text in sorted(by_page.items())]
