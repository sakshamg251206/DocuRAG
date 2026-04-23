# backend/app.py - Enhanced Version
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
import httpx
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import uuid
import shutil
from typing import List, Optional, Dict
import json
import time
import logging
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pdf-chatbot")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="PDF Chatbot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    pdf_id: str
    conversation_history: Optional[List[Dict[str, str]]] = []
    stream: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    response_time_ms: int

class OllamaStatus(BaseModel):
    running: bool
    models: List[str]

# ── State ─────────────────────────────────────────────────────────────────────
pdf_processors: Dict[str, dict] = {}
temp_dir = "temp_pdfs"
os.makedirs(temp_dir, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_PDF_SIZE_MB  = int(os.getenv("MAX_PDF_SIZE_MB", "50"))

# ── Helpers ───────────────────────────────────────────────────────────────────
def check_ollama() -> OllamaStatus:
    import requests as req
    try:
        r = req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return OllamaStatus(running=True, models=models)
    except Exception:
        pass
    return OllamaStatus(running=False, models=[])

def build_prompt_with_history(context: str, query: str, history: List[Dict[str, str]]) -> str:
    """Build a prompt that includes conversation history for context."""
    history_text = ""
    if history:
        history_text = "CONVERSATION HISTORY:\n"
        for turn in history[-6:]:          # keep last 6 turns to stay within context
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"
        history_text += "\n"

    return f"""You are a helpful assistant that answers questions strictly based on the provided PDF context.

{history_text}DOCUMENT CONTEXT:
{context}

CURRENT QUESTION:
{query}

FORMAT REQUIREMENTS:
- Write in clear, well-structured paragraphs.
- Use "- " bullet points when listing items.
- Use two line breaks between paragraphs.
- If the answer is not in the context, say so honestly — do not fabricate information.

ANSWER:"""

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    status = check_ollama()
    if not status.running:
        logger.warning("Ollama is NOT running. Start it with: ollama serve")
    else:
        logger.info(f"Ollama running. Available models: {status.models}")

# ── Health & Info ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "PDF Chatbot API v2.0 is running", "docs": "/docs"}

@app.get("/health")
async def health():
    ollama = check_ollama()
    return {
        "status": "ok",
        "ollama": ollama.dict(),
        "loaded_pdfs": len([p for p in pdf_processors.values() if p.get("status") == "ready"]),
        "timestamp": datetime.utcnow().isoformat(),
    }

# ── PDF Management ────────────────────────────────────────────────────────────
@app.post("/upload-pdf/")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Size guard
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"PDF exceeds {MAX_PDF_SIZE_MB} MB limit.")

    pdf_id = str(uuid.uuid4())
    file_path = os.path.join(temp_dir, f"{pdf_id}.pdf")

    with open(file_path, "wb") as f:
        f.write(contents)

    pdf_processors[pdf_id] = {
        "filename": file.filename,
        "status": "processing",
        "uploaded_at": datetime.utcnow().isoformat(),
        "size_mb": round(size_mb, 2),
    }

    logger.info(f"Received PDF '{file.filename}' ({size_mb:.1f} MB) → id={pdf_id}")
    background_tasks.add_task(process_pdf, file_path, pdf_id, file.filename)

    return {"pdf_id": pdf_id, "filename": file.filename, "status": "processing"}


@app.get("/pdfs/")
async def list_pdfs():
    """Return all loaded PDFs and their metadata."""
    return [
        {
            "pdf_id": pid,
            "filename": data.get("filename"),
            "status": data.get("status"),
            "uploaded_at": data.get("uploaded_at"),
            "size_mb": data.get("size_mb"),
            "pages": data.get("pages"),
            "chunks": data.get("chunks"),
        }
        for pid, data in pdf_processors.items()
    ]


@app.get("/pdf-status/{pdf_id}")
async def pdf_status(pdf_id: str):
    if pdf_id not in pdf_processors:
        raise HTTPException(status_code=404, detail="PDF not found.")
    data = pdf_processors[pdf_id]
    return {
        "pdf_id": pdf_id,
        "filename": data.get("filename", "Unknown"),
        "status": data.get("status", "processing"),
        "pages": data.get("pages"),
        "chunks": data.get("chunks"),
        "error": data.get("error"),
    }


@app.delete("/pdf/{pdf_id}")
async def delete_pdf(pdf_id: str):
    if pdf_id not in pdf_processors:
        raise HTTPException(status_code=404, detail="PDF not found.")
    del pdf_processors[pdf_id]
    logger.info(f"Deleted PDF id={pdf_id}")
    return {"message": f"PDF {pdf_id} deleted successfully."}


# ── Query Endpoints ───────────────────────────────────────────────────────────
@app.post("/query/", response_model=QueryResponse)
async def query(request: QueryRequest):
    _validate_pdf_ready(request.pdf_id)
    t0 = time.time()
    try:
        retriever = pdf_processors[request.pdf_id]["retriever"]
        qa = RetrievalQA.from_chain_type(
            llm=Ollama(model=LLM_MODEL, temperature=0.1, base_url=OLLAMA_BASE_URL),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        result = qa({"query": request.query})
        sources = [f"Page {doc.metadata.get('page', '?')}" for doc in result.get("source_documents", [])]
        elapsed = int((time.time() - t0) * 1000)
        logger.info(f"Query answered in {elapsed}ms for pdf_id={request.pdf_id}")
        return {"answer": result["result"], "sources": sources, "response_time_ms": elapsed}
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying PDF: {str(e)}")


@app.post("/query-stream/")
async def query_stream(request: QueryRequest):
    _validate_pdf_ready(request.pdf_id)
    try:
        retriever = pdf_processors[request.pdf_id]["retriever"]
        docs = retriever.get_relevant_documents(request.query)
        separator = "\n" + "─" * 50 + "\n"
        context = separator.join([doc.page_content for doc in docs])
        sources = list({f"Page {doc.metadata.get('page', '?')}" for doc in docs})

        prompt = build_prompt_with_history(context, request.query, request.conversation_history or [])

        logger.info(f"Streaming response for pdf_id={request.pdf_id} | query='{request.query[:60]}'")

        async def event_stream():
            # First emit sources as a metadata event
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
            async for chunk in stream_llm_response(prompt):
                yield chunk

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Stream error: {e}")
        raise HTTPException(status_code=500, detail=f"Error streaming response: {str(e)}")


# ── Internal ──────────────────────────────────────────────────────────────────
def _validate_pdf_ready(pdf_id: str):
    if pdf_id not in pdf_processors:
        raise HTTPException(status_code=404, detail="PDF not found.")
    if pdf_processors[pdf_id].get("status") != "ready":
        status = pdf_processors[pdf_id].get("status", "unknown")
        raise HTTPException(status_code=400, detail=f"PDF is not ready (status: {status}).")


def process_pdf(file_path: str, pdf_id: str, filename: str):
    try:
        logger.info(f"Processing '{filename}' (id={pdf_id})")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        for i, doc in enumerate(documents):
            doc.metadata["page"] = i + 1

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            keep_separator=True,
            separators=[r"\nQus.\d+", "\n\n", "\n", " "],
            is_separator_regex=True,
        )
        chunks = text_splitter.split_documents(documents)

        logger.info(f"  → {len(documents)} pages, {len(chunks)} chunks")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="mmr",                            # ← Maximum Marginal Relevance (more diverse results)
            search_kwargs={"k": 5, "fetch_k": 20},
        )

        pdf_processors[pdf_id].update({
            "retriever": retriever,
            "status": "ready",
            "pages": len(documents),
            "chunks": len(chunks),
        })
        logger.info(f"PDF '{filename}' ready.")

    except Exception as e:
        logger.error(f"Error processing '{filename}': {e}")
        pdf_processors[pdf_id].update({"status": "error", "error": str(e)})
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


async def stream_llm_response(prompt: str):
    url = f"{OLLAMA_BASE_URL}/api/generate"
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream(
                "POST",
                url,
                headers={"Content-Type": "application/json"},
                json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                yield f"data: {chunk}\n\n"
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            pass
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: ⚠ Error connecting to Ollama: {str(e)}\n\n"
            yield "data: [DONE]\n\n"


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)