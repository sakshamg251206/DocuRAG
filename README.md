# 🧠 RAG PDF Chatbot 

A production-ready Retrieval-Augmented Generation (RAG) chatbot that lets you chat with any PDF document. Built with LangChain, Ollama (Llama 3.2), FastAPI, and Streamlit.

---

| Area | Enhancement |
|---|---|
| **Multi-PDF support** | Upload and switch between multiple PDFs without restarting |
| **Conversation history** | The LLM receives the last 6 turns so follow-up questions work naturally |
| **Source chips** | Retrieved page numbers shown as styled chips under each answer |
| **MMR retrieval** | Switched from plain similarity to Maximum Marginal Relevance for more diverse context chunks |
| **PDF manager** | Sidebar lists all loaded PDFs with status, page count, chunk count, and a delete button |
| **Health endpoint** | `/health` reports API status, Ollama status, available models, and loaded PDF count |
| **PDF list endpoint** | `GET /pdfs/` returns metadata for all loaded documents |
| **Delete endpoint** | `DELETE /pdf/{id}` removes a PDF from memory |
| **File size guard** | Rejects PDFs over 50 MB (configurable via `MAX_PDF_SIZE_MB` env var) |
| **Structured logging** | All backend events logged with timestamps and levels |
| **Environment config** | `OLLAMA_BASE_URL`, `LLM_MODEL`, `EMBED_MODEL`, `MAX_PDF_SIZE_MB` configurable via env |
| **Cursor indicator** | Streaming responses show a `▌` cursor while text is arriving |
| **Timeout handling** | Frontend shows a clear message on slow model responses |
| **Dark UI** | Redesigned frontend with IBM Plex fonts, monospace accents, source chips, and clean dark theme |
| **Dependency cleanup** | Removed duplicate `PyPDF2`, pinned versions, added `python-dotenv` |

---

## 🗂 Project Structure

```
RAG-BASED-CHAT/
├── backend/
│   └── app.py          # FastAPI backend (enhanced)
├── frontend/
│   └── app.py          # Streamlit frontend (enhanced)
├── requirements.txt    # Pinned, compatible dependencies
└── README.md
```

---

## ⚙️ Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com/)** installed and running
3. Llama 3.2 pulled:
   ```bash
   ollama pull llama3.2:3b
   ```
4. Verify Ollama:
   ```bash
   curl http://localhost:11434/api/tags
   ```

---

## 🚀 Installation

```bash
git clone https://github.com/Sakshamg251206/DocuRAG.git
cd DocuRAG

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt --no-cache-dir
```

---

## ▶️ Running

**Terminal 1 — Backend:**
```bash
python backend/app.py
# API docs at http://localhost:8000/docs
# Health check at http://localhost:8000/health
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/app.py
# Open http://localhost:8501
```

---

## 🔧 Environment Variables

Override defaults without touching code:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3.2:3b` | Model used for generation |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `MAX_PDF_SIZE_MB` | `50` | Maximum upload size in MB |

Example:
```bash
LLM_MODEL=llama3.1:8b python backend/app.py
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health + Ollama status |
| `POST` | `/upload-pdf/` | Upload and index a PDF |
| `GET` | `/pdfs/` | List all loaded PDFs |
| `GET` | `/pdf-status/{id}` | Status of a specific PDF |
| `DELETE` | `/pdf/{id}` | Remove a PDF from memory |
| `POST` | `/query/` | Non-streaming Q&A |
| `POST` | `/query-stream/` | Streaming Q&A (SSE) |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔬 How It Works

```
PDF Upload
    │
    ▼
PyPDFLoader → RecursiveCharacterTextSplitter (1000 chars / 200 overlap)
    │
    ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
    │
    ▼
FAISS Vector Store (IndexFlatL2)
    │
    ▼
MMR Retriever (k=5, fetch_k=20)  ← more diverse than plain similarity
    │
User Question + Conversation History
    │
    ▼
Prompt Builder → Ollama (Llama 3.2) → Streaming SSE response
```

---

## ⚠️ Limitations

- Text-based PDFs only (no OCR for scanned documents)
- All state is in-memory — PDFs must be re-uploaded after restart
- Performance depends on your CPU and the Ollama model size

---

## 🗺 Roadmap

- [ ] Persistent vector store (SQLite / ChromaDB)
- [ ] OCR support via `pytesseract`
- [ ] Dynamic LLM switching from the UI
- [ ] Multi-document cross-query
- [ ] Answer confidence scoring
- [ ] Docker Compose deployment