"""DocuRAG web interface.

Run with `streamlit run frontend/app.py`. The API URL is read from `DOCURAG_API_URL`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from typing import Any

import streamlit as st

from api_client import ApiClient, ApiError

API_URL = os.getenv("DOCURAG_API_URL", "http://localhost:8000")
MAX_UPLOAD_MB = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
HISTORY_MESSAGES = 12  # recent messages sent with each question so follow-ups make sense

STARTER_QUESTIONS = [
    "Summarize this document in a few bullet points.",
    "What are the main topics covered?",
    "List any important dates, numbers or deadlines.",
]

STATUS_LABELS = {
    "ready": ":green[● Ready]",
    "processing": ":orange[● Indexing…]",
    "failed": ":red[● Failed]",
}

st.set_page_config(page_title="DocuRAG · Chat with your PDFs", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
      .block-container { max-width: 1000px; padding-top: 2.5rem; }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { line-height: 1.35; }
      blockquote { font-size: 0.9rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── State & helpers ──────────────────────────────────────────────────────────


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient(API_URL)


def init_state() -> None:
    st.session_state.setdefault("active_id", None)
    st.session_state.setdefault("chats", {})  # document id → list of messages
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("pending_question", None)


def md_escape(text: str) -> str:
    """Escape Markdown control characters so user-supplied names render literally."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$:])", r"\\\1", text)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024 or unit == "MB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} MB"


def describe(doc: dict[str, Any]) -> str:
    if doc["status"] == "ready":
        return f"{doc['pages']} pages · {doc['chunks']} passages · {human_size(doc['size_bytes'])}"
    return human_size(doc["size_bytes"])


def chat_for(document_id: str) -> list[dict[str, Any]]:
    chats: dict[str, list[dict[str, Any]]] = st.session_state.chats
    return chats.setdefault(document_id, [])


def select_document(document_id: str) -> None:
    st.session_state.active_id = document_id


# ── Sidebar ──────────────────────────────────────────────────────────────────


def render_upload(client: ApiClient, api_online: bool, location: str, label: str = "Upload a PDF") -> None:
    uploaded = st.file_uploader(
        label,
        type="pdf",
        key=f"uploader_{location}_{st.session_state.uploader_key}",
        max_upload_size=MAX_UPLOAD_MB,
        disabled=not api_online,
        help=f"PDFs with selectable text, up to {MAX_UPLOAD_MB} MB. Scanned images are not supported.",
    )
    if uploaded is None:
        return
    try:
        with st.spinner(f"Uploading {uploaded.name}…"):
            created = client.upload(uploaded.name, uploaded.getvalue())
    except ApiError as exc:
        st.error(str(exc), icon=":material/error:")
        return
    st.session_state.active_id = created["id"]
    st.session_state.uploader_key += 1  # clears the widget so the file isn't uploaded again
    st.toast(f"Uploaded **{md_escape(created['filename'])}**. Indexing has started.", icon="📄")
    st.rerun()


def render_document_list(client: ApiClient, documents: list[dict[str, Any]]) -> None:
    st.markdown("**Your documents**")
    if not documents:
        st.caption("Nothing here yet. Uploaded documents will appear in this list.")
        return

    for doc in documents:
        is_active = doc["id"] == st.session_state.active_id
        with st.container(border=True):
            name_col, menu_col = st.columns([0.8, 0.2], vertical_alignment="center")
            name_col.button(
                md_escape(doc["filename"]),
                key=f"open_{doc['id']}",
                type="primary" if is_active else "tertiary",
                icon=":material/description:",
                on_click=select_document,
                args=(doc["id"],),
                width="stretch",
                help="Open this document",
            )
            with menu_col.popover("", icon=":material/more_vert:", help="Options"):
                st.caption("Remove this document and its chat history?")
                if st.button("Delete", key=f"delete_{doc['id']}", type="primary", icon=":material/delete:"):
                    try:
                        client.delete(doc["id"])
                    except ApiError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.chats.pop(doc["id"], None)
                        if is_active:
                            st.session_state.active_id = None
                        st.rerun()
            st.caption(f"{STATUS_LABELS.get(doc['status'], doc['status'])} · {describe(doc)}")


def render_system_status(health: dict[str, Any] | None) -> None:
    with st.expander("System status", icon=":material/monitor_heart:", expanded=health is None):
        if health is None:
            st.markdown(":red[● API offline]")
            st.caption(f"Expected at `{API_URL}`")
            return
        llm = health["llm"]
        st.markdown(":green[● API online]")
        if not llm["reachable"]:
            st.markdown(":red[● Ollama offline]")
        elif not llm["model_available"]:
            st.markdown(":orange[● Model not installed]")
        else:
            st.markdown(":green[● Ollama ready]")
        st.caption(
            f"LLM: `{llm['model']}`  \nEmbeddings: `{health['embedding_model']}`  \nAPI version {health['version']}"
        )


# ── Main area ────────────────────────────────────────────────────────────────


def render_welcome(client: ApiClient, has_documents: bool, api_online: bool) -> None:
    st.title("Chat with your PDFs")
    st.markdown(
        "DocuRAG answers questions about your documents **using only what is written in them**, "
        "and shows the pages each answer came from. Everything runs on your own machine: "
        "your files are never sent to a cloud service."
    )
    st.space("small")
    steps = [
        (":material/upload_file:", "1. Upload a PDF", "Add a manual, report, policy, contract or FAQ."),
        (":material/manage_search:", "2. Let it index", "The text is split into passages and indexed for search."),
        (":material/forum:", "3. Ask questions", "Get answers grounded in the document, with page citations."),
    ]
    for column, (icon, title, text) in zip(st.columns(3), steps, strict=True):
        with column.container(border=True, height="stretch"):
            st.markdown(f"#### {icon} {title}")
            st.caption(text)
    st.space("small")
    render_upload(client, api_online, location="main", label="Choose a PDF to get started")
    if has_documents:
        st.caption(":material/history: Or open one of your documents from the sidebar.")
    else:
        st.caption(":material/lightbulb: No PDF at hand? Try `examples/sample-faq.pdf` from the project repository.")


def render_service_warnings(health: dict[str, Any]) -> None:
    llm = health["llm"]
    if not llm["reachable"]:
        st.warning(
            f"**The language model is offline.** Documents can still be uploaded, but answers need "
            f"Ollama running at `{llm['base_url']}`. Start it with `ollama serve`.",
            icon=":material/power_off:",
        )
    elif not llm["model_available"]:
        st.warning(
            f"**The model `{llm['model']}` is not installed.** Run `ollama pull {llm['model']}` and refresh.",
            icon=":material/download:",
        )


@st.fragment(run_every=2)
def watch_indexing(client: ApiClient, document_id: str) -> None:
    """Poll a document that is being indexed and refresh the page once it's done."""
    try:
        doc = client.get_document(document_id)
    except ApiError:
        doc = None
    if doc is None or doc["status"] != "processing":
        st.rerun(scope="app")
    with st.container(border=True):
        st.markdown("**:material/hourglass_top: Indexing your document…**")
        st.caption(
            "Extracting the text, splitting it into passages and computing embeddings. "
            "This usually takes a few seconds; large documents can take a minute. "
            "This page updates automatically."
        )


def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user" else "📄"):
        if message.get("error"):
            st.error(message["content"], icon=":material/error:")
        else:
            st.markdown(message["content"])
        render_sources(message.get("sources") or [])


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    pages = ", ".join(str(s["page"]) for s in sources)
    label = f"Sources · page {pages}" if len(sources) == 1 else f"Sources · pages {pages}"
    with st.expander(label, icon=":material/menu_book:"):
        for source in sources:
            st.markdown(f"**Page {source['page']}**\n\n> {md_escape(source['excerpt'])}")


def answer(client: ApiClient, document_id: str, question: str) -> None:
    messages = chat_for(document_id)
    history = [{"role": m["role"], "content": m["content"]} for m in messages[-HISTORY_MESSAGES:] if not m.get("error")]
    messages.append({"role": "user", "content": question})
    render_message(messages[-1])

    result: dict[str, Any] = {"sources": [], "error": None}
    with st.chat_message("assistant", avatar="📄"):
        thinking = st.empty()
        thinking.caption(":material/manage_search: Searching the document and composing an answer…")

        def tokens() -> Iterator[str]:
            for event in client.ask_stream(document_id, question, history):
                if event.event == "sources":
                    result["sources"] = event.data
                elif event.event == "token":
                    thinking.empty()
                    yield event.data
                elif event.event == "error":
                    result["error"] = event.data.get("detail", "Something went wrong.")
                    return
                elif event.event == "done":
                    return

        try:
            streamed = st.write_stream(tokens(), cursor="▌")
            text = streamed if isinstance(streamed, str) else "".join(map(str, streamed))
        except ApiError as exc:
            text, result["error"] = "", str(exc)
        thinking.empty()

        if result["error"]:
            st.error(result["error"], icon=":material/error:")
            messages.append({"role": "assistant", "content": result["error"], "error": True})
            return
        if not text.strip():
            text = "_The model returned an empty answer. Try rephrasing your question._"
            st.markdown(text)
        render_sources(result["sources"])
    messages.append({"role": "assistant", "content": text, "sources": result["sources"]})


def render_chat(client: ApiClient, doc: dict[str, Any], llm_ready: bool) -> None:
    messages = chat_for(doc["id"])

    # The chat input is pinned to the bottom of the page, so reading it first doesn't move it, and
    # lets this run render the conversation (rather than the empty state) as soon as a question is sent.
    typed = st.chat_input(
        "Ask a question about this document…" if llm_ready else "Waiting for the language model…",
        disabled=not llm_ready,
        max_chars=4000,
    )
    pending = st.session_state.pending_question
    st.session_state.pending_question = None
    question = (typed if isinstance(typed, str) else pending or "").strip()

    title_col, action_col = st.columns([0.8, 0.2], vertical_alignment="bottom")
    title_col.subheader(f":material/description: {md_escape(doc['filename'])}", anchor=False)
    title_col.caption(describe(doc))
    if messages and action_col.button("New chat", icon=":material/refresh:", width="stretch"):
        messages.clear()
        st.rerun()

    if not messages and not question:
        with st.container(border=True):
            st.markdown("**Ask anything about this document.**")
            st.caption("Answers are based only on the document's content. Start with one of these:")
            with st.container(horizontal=True):
                for i, suggestion in enumerate(STARTER_QUESTIONS):
                    st.button(
                        suggestion,
                        key=f"starter_{i}",
                        disabled=not llm_ready,
                        on_click=st.session_state.__setitem__,
                        args=("pending_question", suggestion),
                    )

    for message in messages:
        render_message(message)

    if question:
        answer(client, doc["id"], question)


def render_document(client: ApiClient, doc: dict[str, Any], llm_ready: bool) -> None:
    if doc["status"] == "processing":
        st.subheader(f":material/description: {md_escape(doc['filename'])}", anchor=False)
        watch_indexing(client, doc["id"])
    elif doc["status"] == "failed":
        st.subheader(f":material/description: {md_escape(doc['filename'])}", anchor=False)
        st.error(f"**This document could not be indexed.** {doc.get('error') or ''}", icon=":material/error:")
        st.caption("Delete it from the sidebar and try a different file.")
    else:
        render_chat(client, doc, llm_ready)


# ── Page ─────────────────────────────────────────────────────────────────────


def main() -> None:
    init_state()
    client = get_client()

    try:
        health: dict[str, Any] | None = client.health()
        documents = client.list_documents()
    except ApiError:
        health, documents = None, []

    by_id = {doc["id"]: doc for doc in documents}
    if st.session_state.active_id not in by_id:
        st.session_state.active_id = None

    with st.sidebar:
        st.markdown("## 📄 DocuRAG")
        st.caption("Private question answering over your PDFs, powered by a local LLM.")
        if st.session_state.active_id in by_id:
            render_upload(client, api_online=health is not None, location="sidebar")
        render_document_list(client, documents)
        st.space("medium")
        render_system_status(health)

    if health is None:
        render_welcome(client, has_documents=False, api_online=False)
        st.error(
            f"**Can't reach the DocuRAG API at `{API_URL}`.** Start it with `make api` "
            "(or `uvicorn backend.main:app`) and refresh this page.",
            icon=":material/cloud_off:",
        )
        return

    llm = health["llm"]
    llm_ready = bool(llm["reachable"] and llm["model_available"])
    active = by_id.get(st.session_state.active_id) if st.session_state.active_id else None
    if active is None:
        render_welcome(client, has_documents=bool(documents), api_online=True)
        render_service_warnings(health)
        return

    render_service_warnings(health)
    render_document(client, active, llm_ready)


main()
