import streamlit as st
import requests
import time
import os

# API Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PDF Chatbot with LangChain",
    page_icon="📄",
    layout="wide"
)

def main():
    st.title("📄 PDF Chatbot with LangChain & Llama 3.2")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_id" not in st.session_state:
        st.session_state.pdf_id = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "uploaded_pdf_name" not in st.session_state:
        st.session_state.uploaded_pdf_name = None
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload PDF")
        pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if pdf_file and pdf_file.name != st.session_state.uploaded_pdf_name:
            st.session_state.processing = True
            st.session_state.uploaded_pdf_name = pdf_file.name
            with st.spinner("Processing PDF..."):
                # Upload the PDF only if it's a new file
                response = requests.post(
                    f"{API_URL}/upload-pdf/", 
                    files={"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.pdf_id = data["pdf_id"]
                    
                    # Wait for processing to complete
                    complete = False
                    attempts = 0
                    while not complete and attempts < 30:
                        status_response = requests.get(f"{API_URL}/pdf-status/{st.session_state.pdf_id}")
                        if status_response.status_code == 200:
                            status_data = status_response.json()
                            if status_data["status"] == "ready":
                                complete = True
                                st.success(f"PDF processed: {status_data['filename']}")
                                st.session_state.messages = []  # Reset chat on new upload
                            elif status_data["status"] == "error":
                                st.error(f"Error processing PDF: {status_data.get('error', 'Unknown error')}")
                                st.session_state.pdf_id = None
                                break
                            else:
                                time.sleep(1)
                        attempts += 1
                    
                    if not complete:
                        st.warning("PDF processing is taking longer than expected. You can try asking questions, but the system might not be ready yet.")
                else:
                    st.error(f"Error uploading PDF: {response.text}")
                    st.session_state.uploaded_pdf_name = None
            
            st.session_state.processing = False
            
        if st.session_state.pdf_id:
            st.info("You can now ask questions about the PDF in the chat.")
            if st.button("Reset Chat"):
                st.session_state.messages = []
    
    # Display chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message and message["sources"]:
                st.caption(f"Sources: {', '.join(message['sources'])}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the PDF..."):
        if not st.session_state.pdf_id:
            st.info("Please upload a PDF first.")
            return
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Inside your chat input handler, replace the existing streaming code with this:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            sources = []
            
            try:
                response = requests.post(
                    f"{API_URL}/query-stream/",
                    json={"query": prompt, "pdf_id": st.session_state.pdf_id, "stream": True},
                    stream=True
                )
                
                if response.status_code == 200:
                    # Initialize with a loading message
                    message_placeholder.markdown("Thinking...")
                    
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                chunk = line[6:]
                                if chunk == "[DONE]":
                                    break
                                
                                # Add the new chunk to our response
                                full_response += chunk
            
                                message_placeholder.markdown(full_response)
                else:
                    error_msg = f"Error: {response.text}"
                    message_placeholder.error(error_msg)
                    full_response = error_msg
            except Exception as e:
                error_msg = f"Error connecting to the API: {str(e)}"
                message_placeholder.error(error_msg)
                full_response = error_msg
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": sources
            })

if __name__ == "__main__":
    main()