# frontend/app.py - Enhanced Version
import streamlit as st
import requests
import time
import json

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"
    
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Background & Base ── */
.stApp {
    background: #0d0f14;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2230;
}
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

/* ── Header ── */
.main-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.4rem 0 0.6rem;
    border-bottom: 1px solid #1e2230;
    margin-bottom: 1.2rem;
}
.main-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 600;
    color: #7dd3fc;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header .subtitle {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 2px;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #13161f !important;
    border: 1px solid #1e2230 !important;
    border-radius: 10px !important;
    padding: 0.9rem 1.1rem !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #0f1720 !important;
    border-color: #1e3a5f !important;
}

/* ── Source chips ── */
.source-chip {
    display: inline-block;
    background: #1a2540;
    border: 1px solid #2a4070;
    color: #7dd3fc;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px 3px 2px 0;
}

/* ── Status badges ── */
.badge-ready    { color: #4ade80; font-size: 0.72rem; }
.badge-processing { color: #facc15; font-size: 0.72rem; }
.badge-error    { color: #f87171; font-size: 0.72rem; }

/* ── PDF card ── */
.pdf-card {
    background: #13161f;
    border: 1px solid #1e2230;
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: border-color 0.15s;
}
.pdf-card.active { border-color: #7dd3fc; }
.pdf-card:hover  { border-color: #334155; }
.pdf-card .pdf-name {
    font-size: 0.82rem;
    font-weight: 500;
    color: #e2e8f0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pdf-card .pdf-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #475569;
    margin-top: 2px;
}

/* ── Input ── */
[data-testid="stChatInput"] textarea {
    background: #13161f !important;
    border: 1px solid #2a3548 !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #1e2a3a !important;
    border: 1px solid #2a3f5f !important;
    color: #7dd3fc !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #243447 !important;
    border-color: #7dd3fc !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #13161f;
    border: 1px solid #1e2230;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #7dd3fc !important; font-size: 1.3rem !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #7dd3fc !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #1e2a3a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "messages": [],           # list of {role, content, sources}
    "pdf_id": None,
    "pdf_name": None,
    "uploaded_pdf_name": None,
    "processing": False,
    "sources_latest": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── API Helpers ───────────────────────────────────────────────────────────────
def api_get(path, **kwargs):
    try:
        return requests.get(f"{API_URL}{path}", timeout=10, **kwargs)
    except requests.exceptions.ConnectionError:
        return None

def api_post(path, **kwargs):
    try:
        return requests.post(f"{API_URL}{path}", timeout=10, **kwargs)
    except requests.exceptions.ConnectionError:
        return None

def api_delete(path):
    try:
        return requests.delete(f"{API_URL}{path}", timeout=10)
    except requests.exceptions.ConnectionError:
        return None

def get_health():
    r = api_get("/health")
    if r and r.status_code == 200:
        return r.json()
    return None

def get_pdfs():
    r = api_get("/pdfs/")
    if r and r.status_code == 200:
        return r.json()
    return []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RAG PDF Chat")
    st.markdown("---")

    # Health status
    health = get_health()
    if health:
        ollama_ok = health.get("ollama", {}).get("running", False)
        col1, col2 = st.columns(2)
        col1.metric("API", "● Online", delta_color="off")
        col2.metric("Ollama", "● Online" if ollama_ok else "● Offline", delta_color="off")
        if ollama_ok:
            models = health.get("ollama", {}).get("models", [])
            if models:
                st.caption(f"Models: {', '.join(models[:3])}")
    else:
        st.error("⚠ Cannot reach API server")

    st.markdown("---")
    st.markdown("### 📤 Upload PDF")

    pdf_file = st.file_uploader("Choose a PDF", type="pdf", label_visibility="collapsed")

    if pdf_file and pdf_file.name != st.session_state.uploaded_pdf_name:
        st.session_state.processing = True
        st.session_state.uploaded_pdf_name = pdf_file.name

        with st.spinner(f"Uploading & indexing {pdf_file.name}…"):
            r = api_post(
                "/upload-pdf/",
                files={"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")},
            )
            if r and r.status_code == 200:
                pdf_id = r.json()["pdf_id"]

                # Poll for readiness
                for _ in range(60):
                    sr = api_get(f"/pdf-status/{pdf_id}")
                    if sr and sr.status_code == 200:
                        sd = sr.json()
                        if sd["status"] == "ready":
                            st.session_state.pdf_id = pdf_id
                            st.session_state.pdf_name = pdf_file.name
                            st.session_state.messages = []
                            st.success(f"✓ {sd['pages']} pages, {sd['chunks']} chunks")
                            break
                        elif sd["status"] == "error":
                            st.error(f"Processing failed: {sd.get('error', 'unknown')}")
                            st.session_state.uploaded_pdf_name = None
                            break
                    time.sleep(1)
                else:
                    st.warning("Still processing — try asking questions in a moment.")
            else:
                st.error("Upload failed. Is the backend running?")
                st.session_state.uploaded_pdf_name = None

        st.session_state.processing = False

    # ── PDF list ──
    st.markdown("---")
    st.markdown("### 📚 Loaded PDFs")

    pdfs = get_pdfs()
    if not pdfs:
        st.caption("No PDFs loaded yet.")
    else:
        for pdf in pdfs:
            pid   = pdf["pdf_id"]
            name  = pdf.get("filename", "Unknown")
            status = pdf.get("status", "?")
            pages  = pdf.get("pages", "?")
            chunks = pdf.get("chunks", "?")
            is_active = pid == st.session_state.pdf_id

            badge_cls = {"ready": "badge-ready", "processing": "badge-processing"}.get(status, "badge-error")
            badge_sym = {"ready": "●", "processing": "◌"}.get(status, "✕")

            col_a, col_b = st.columns([5, 1])
            with col_a:
                card_cls = "pdf-card active" if is_active else "pdf-card"
                st.markdown(
                    f"""<div class="{card_cls}">
                        <div class="pdf-name">{'▶ ' if is_active else ''}{name}</div>
                        <div class="pdf-meta">
                            <span class="{badge_cls}">{badge_sym} {status}</span>
                            &nbsp;·&nbsp;{pages}pp · {chunks} chunks
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if status == "ready" and not is_active:
                    if st.button("Switch", key=f"sw_{pid}"):
                        st.session_state.pdf_id   = pid
                        st.session_state.pdf_name = name
                        st.session_state.messages = []
                        st.rerun()
            with col_b:
                if st.button("🗑", key=f"del_{pid}"):
                    api_delete(f"/pdf/{pid}")
                    if pid == st.session_state.pdf_id:
                        st.session_state.pdf_id   = None
                        st.session_state.pdf_name = None
                        st.session_state.messages = []
                    st.rerun()

    st.markdown("---")
    if st.session_state.pdf_id and st.button("🗑 Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown(
    """<div class="main-header">
        <div>
            <h1>RAG PDF Chatbot</h1>
            <div class="subtitle">Retrieval-Augmented Generation · Llama 3.2 · LangChain</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# Active PDF banner
if st.session_state.pdf_id and st.session_state.pdf_name:
    st.info(f"📄 Active document: **{st.session_state.pdf_name}**", icon="📄")
else:
    st.markdown(
        """<div style='text-align:center;padding:3rem 1rem;color:#475569'>
            <div style='font-size:3rem'>📂</div>
            <div style='font-size:1rem;margin-top:0.5rem'>Upload a PDF in the sidebar to begin</div>
        </div>""",
        unsafe_allow_html=True,
    )

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            chips = "".join(f'<span class="source-chip">{s}</span>' for s in msg["sources"])
            st.markdown(f'<div style="margin-top:6px">{chips}</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask something about the PDF…",
    disabled=not st.session_state.pdf_id,
):
    if not st.session_state.pdf_id:
        st.warning("Please upload a PDF first.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation history (last 10 turns)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-10:]
        if m["role"] in ("user", "assistant")
    ]

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder   = st.empty()
        sources_spot  = st.empty()
        full_response = ""
        sources: list = []

        try:
            resp = requests.post(
                f"{API_URL}/query-stream/",
                json={
                    "query": prompt,
                    "pdf_id": st.session_state.pdf_id,
                    "stream": True,
                    "conversation_history": history,
                },
                stream=True,
                timeout=120,
            )

            if resp.status_code == 200:
                placeholder.markdown("_Thinking…_")

                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")

                    # Sources metadata event
                    if decoded.startswith("event: sources"):
                        continue
                    if decoded.startswith("data: ") and not full_response:
                        # Try to parse as sources JSON first
                        raw = decoded[6:]
                        try:
                            maybe = json.loads(raw)
                            if isinstance(maybe, list):
                                sources = maybe
                                if sources:
                                    chips = "".join(f'<span class="source-chip">{s}</span>' for s in sources)
                                    sources_spot.markdown(
                                        f'<div style="margin-bottom:6px">{chips}</div>',
                                        unsafe_allow_html=True,
                                    )
                                continue
                        except (json.JSONDecodeError, TypeError):
                            pass

                    if decoded.startswith("data: "):
                        chunk = decoded[6:]
                        if chunk == "[DONE]":
                            break
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

            else:
                err = f"⚠ API error ({resp.status_code}): {resp.text}"
                placeholder.error(err)
                full_response = err

        except requests.exceptions.Timeout:
            err = "⚠ Request timed out. The model might be slow — try again."
            placeholder.error(err)
            full_response = err
        except Exception as e:
            err = f"⚠ Connection error: {str(e)}"
            placeholder.error(err)
            full_response = err

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })