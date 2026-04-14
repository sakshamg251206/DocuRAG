# PDF Chatbot with LangChain and Llama 3.2

This repository contains a question-answering chatbot that uses a PDF document as its knowledge base. It leverages LangChain, Ollama running Llama 3.2, FastAPI, and Streamlit to provide an end-to-end solution for interacting with PDF content.

## Features

- Upload and process PDF documents
- Ask questions about the PDF content
- Retrieve accurate answers with citations to source pages
- Streaming responses for a better user experience
- Simple and intuitive UI

## Technologies Used

- **LangChain**: Orchestrates document processing and retrieval-augmented generation (RAG)
- **Ollama**: Runs Llama 3.2 locally for inference
- **FastAPI**: Provides the backend API
- **Streamlit**: Delivers a clean, interactive frontend
- **FAISS**: Enables efficient vector similarity search
- **HuggingFace Sentence Transformers**: Powers text embeddings

## Prerequisites

1. [Ollama](https://ollama.com/) installed locally
2. Llama 3.2 model pulled in Ollama:
   ```bash
   ollama pull llama3.2
   ```
3. Python 3.8+ installed
4. Verify that Ollama is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```
   - If it returns a list of models, Ollama is running.
   - If you get a connection error, start Ollama manually:
     ```bash
     ollama serve
     ```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/CoderArnav-bot/RAG-BASED-CHAT.git
   cd RAG-BASED-CHAT
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   ```

   ```bash
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt --no-cache-dir
   ```

## Running the Application

1. Start the FastAPI backend:
   ```bash
   python backend\app.py
   ```

2. Start the Streamlit frontend in a new terminal:
   ```bash
   cd frontend
   streamlit run frontend\app.py
   ```

3. Open your browser and navigate to [http://localhost:8501](http://localhost:8501)

## Usage

1. Upload a PDF document using the sidebar.
2. Wait for the processing to complete (indicated by a success message).
3. Ask questions in the chat input at the bottom of the page.
4. View AI-generated answers with page citations.

## How It Works

1. **Document Processing**:
   - The PDF text is extracted and split into chunks with overlap.
   - Each chunk is embedded and stored in a FAISS vector store.

2. **Question Answering**:
   - The user's question is embedded.
   - The most relevant chunks are retrieved from the vector store.
   - Context and question are sent to Llama 3.2.
   - An answer is generated and displayed with citations.

## Limitations

- Works best with text-based PDFs (not scanned documents)
- Processing large PDFs may take some time
- Answer quality depends on the Llama 3.2 model capabilities

## Future Improvements

- Adding logs for each request
- Dynamic switching between LLMs
- OCR support for scanned documents
- Multi-document support
- Filter responses by document sections
- Improved answer validation

## Demo
https://github.com/user-attachments/assets/779359ca-855f-4164-8a6d-a0b1f71c82b6

- Tested on cpu based low end pc.

