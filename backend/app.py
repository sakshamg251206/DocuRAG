# backend/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
import httpx
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import uuid
import shutil
from typing import List
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2

app = FastAPI(title="PDF Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    query: str
    pdf_id: str
    stream: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# Global variables
pdf_processors = {}
temp_dir = "temp_pdfs"

# Create temp directory if it doesn't exist
os.makedirs(temp_dir, exist_ok=True)

# Check if Ollama is running
def check_ollama():
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

@app.on_event("startup")
async def startup_event():
    if not check_ollama():
        print("WARNING: Ollama is not running. Please start Ollama.")

@app.get("/")
async def root():
    return {"message": "PDF Chatbot API is running"}

@app.post("/upload-pdf/")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a PDF file using LangChain"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate a unique ID for this PDF
    pdf_id = str(uuid.uuid4())
    
    # Save the uploaded file
    file_path = os.path.join(temp_dir, f"{pdf_id}.pdf")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Process the PDF in the background
    background_tasks.add_task(process_pdf, file_path, pdf_id, file.filename)
    
    return {"pdf_id": pdf_id, "filename": file.filename, "status": "processing"}

@app.get("/pdf-status/{pdf_id}")
async def pdf_status(pdf_id: str):
    if pdf_id not in pdf_processors:
        raise HTTPException(status_code=404, detail="PDF not found")

    pdf_data = pdf_processors[pdf_id]
    
    return {
        "pdf_id": pdf_id,
        "filename": pdf_data.get("filename", "Unknown"),
        "status": pdf_data.get("status", "processing"),
    }


@app.post("/query/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the chatbot about the PDF content"""
    if request.pdf_id not in pdf_processors or pdf_processors[request.pdf_id].get("status") != "ready":
        raise HTTPException(status_code=404, detail="PDF not found or still processing")
    
    try:
        # Get the retriever
        retriever = pdf_processors[request.pdf_id]["retriever"]
        
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=Ollama(model="llama3.2:3b", temperature=0.1),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get answer
        result = qa({"query": request.query})
        
        # Extract source page numbers
        sources = [f"Page {doc.metadata.get('page', 'unknown')}" for doc in result.get("source_documents", [])]
        
        return {"answer": result["result"], "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying PDF: {str(e)}")

@app.post("/query-stream/")
async def query_stream(request: QueryRequest):
    """Stream query results"""
    if request.pdf_id not in pdf_processors or pdf_processors[request.pdf_id].get("status") != "ready":
        raise HTTPException(status_code=404, detail="PDF not found or still processing")

    try:
        # Get documents from retriever
        retriever = pdf_processors[request.pdf_id]["retriever"]
        docs = retriever.get_relevant_documents(request.query)

        # Extract context from documents
        separator = "\n" + "-"*50 + "\n"
        context = separator.join([doc.page_content for doc in docs])

        prompt = f"""You are a helpful assistant that answers questions strictly based on the provided context.
CONTEXT:
{context}

TASK: 
{request.query}

FORMAT REQUIREMENTS:
- Format your answer with proper paragraphs
- Use two line breaks between paragraphs
- Use bullet points with "- " prefix when appropriate
- Make sure all formatting is clearly visible in your plain text response

NOTE: - If the answer cannot be derived from the context, say that you don't know based on the provided information. Don't make up answers that aren't supported by the context.
      - If relevant context is found, please analyse and provide answer in proper readable format.
"""

        return StreamingResponse(
            stream_llm_response(prompt),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error streaming response: {str(e)}")


async def process_pdf(file_path: str, pdf_id: str, filename: str):
    """Process a PDF file using LangChain components"""
    try:
        # Store filename
        pdf_processors[pdf_id] = {"filename": filename}
        
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add page numbers to metadata
        for i, doc in enumerate(documents):
            doc.metadata["page"] = i + 1
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            keep_separator=True,
            separators=[r"\nQus.\d+"],
            is_separator_regex=True
        )
        chunks = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs = {'device': 'cpu'},encode_kwargs = {'normalize_embeddings': True})
        
        vector_store = FAISS(
        embedding_function=embeddings,
        index=IndexFlatL2(len(embeddings.embed_query(" "))),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy="COSINE").from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        pdf_processors[pdf_id]["retriever"] = retriever  # Store reference only
        pdf_processors[pdf_id]["status"] = "ready"
        
        # Clean up
        try:
            os.remove(file_path)
        except:
            pass
            
    except Exception as e:
        pdf_processors[pdf_id] = {"filename": filename, "status": "error", "error": str(e)}
        print(f"Error processing PDF: {e}")

async def stream_llm_response(prompt: str):
    """Stream responses from Ollama asynchronously"""
    url = "http://localhost:11434/api/generate"

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream(
                "POST",
                url,
                headers={"Content-Type": "application/json"},
                json={"model": "llama3.2:3b", "prompt": prompt, "stream": True}
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            json_line = json.loads(line)
                            chunk = json_line.get("response", "")
                            if chunk:
                                # Just send the chunk directly, preserve its formatting
                                yield f"data: {chunk}\n\n"
                            
                            if json_line.get("done", False):
                                break
                        except json.JSONDecodeError:
                            pass
                yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error connecting to Ollama: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)