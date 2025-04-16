import os
import streamlit as st
import time
from pathlib import Path

from src.config import APP_TITLE, APP_DESCRIPTION
from src.document_processing.loader import load_pdf_documents, split_documents
from src.document_processing.processor import enhance_documents
from src.rag.vectorstore import create_vectorstore, save_vectorstore, load_vectorstore
from src.rag.retriever import create_retriever, retrieve_documents
from src.chains.qa_chain import create_custom_qa_chain, extract_sources_from_docs
from src.utils.helpers import format_response, save_uploaded_file, extract_json_from_response

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Helper function to get or create vectorstore
def get_or_create_vectorstore():
    vector_store_path = "vectorstore"
    
    if os.path.exists(vector_store_path) and os.path.isdir(vector_store_path):
        try:
            return load_vectorstore(vector_store_path)
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return None
    else:
        st.info("No vector store found. Please upload PDF documents to create one.")
        return None

# App title and description
st.title(f"🩺 {APP_TITLE}")
st.markdown(APP_DESCRIPTION)

# Sidebar for document upload and settings
with st.sidebar:
    st.header("📚 Document Management")
    
    # Document upload
    uploaded_files = st.file_uploader(
        "Upload diabetes-related PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        process_button = st.button("Process Documents")
        
        if process_button:
            with st.spinner("Processing documents..."):
                # Save uploaded files
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = save_uploaded_file(uploaded_file)
                    file_paths.append(file_path)
                    st.session_state.uploaded_files.append({
                        "name": uploaded_file.name,
                        "path": file_path
                    })
                
                # Load and process documents
                documents = load_pdf_documents(file_paths)
                
                if documents:
                    # Split and enhance documents
                    document_chunks = split_documents(documents)
                    enhanced_chunks = enhance_documents(document_chunks)
                    
                    # Create or update vector store
                    vectorstore = create_vectorstore(enhanced_chunks)
                    save_vectorstore(vectorstore)
                    
                    # Create retriever
                    retriever = create_retriever(vectorstore)
                    st.session_state.retriever = retriever
                    st.session_state.vectorstore_ready = True
                    
                    st.success(f"Successfully processed {len(documents)} documents ({len(enhanced_chunks)} chunks)")
                else:
                    st.error("No documents were successfully loaded. Please check the files and try again.")
    
    # Advanced settings
    st.header("⚙️ Settings")
    
    prompt_type = st.selectbox(
        "Response Style",
        ["standard", "few_shot", "structured"],
        format_func=lambda x: {
            "standard": "Standard Response",
            "few_shot": "Enhanced with Examples",
            "structured": "Structured Output"
        }[x]
    )
    
    st.divider()
    
    # Display uploaded documents
    if st.session_state.uploaded_files:
        st.header("📄 Uploaded Documents")
        for file in st.session_state.uploaded_files:
            st.write(f"- {file['name']}")
    
    st.divider()
    
    # Credits
    st.markdown("### About")
    st.markdown("""
    Diabetes Management Assistant is a RAG-based medical chatbot powered by:
    - Google Gemini API
    - LangChain
    - FAISS Vector Database
    - Streamlit
    """)

# Initialize vectorstore at startup if not initialized
if not st.session_state.vectorstore_ready:
    vectorstore = get_or_create_vectorstore()
    if vectorstore:
        retriever = create_retriever(vectorstore)
        st.session_state.retriever = retriever
        st.session_state.vectorstore_ready = True

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about diabetes management..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        if not st.session_state.vectorstore_ready:
            st.markdown("Please upload and process documents before asking questions.")
        else:
            # Create a placeholder for the response
            response_placeholder = st.empty()
            sources_container = st.container()
            
            # Create thinking spinner
            with st.spinner("Thinking..."):
                # Retrieve relevant documents
                docs = retrieve_documents(st.session_state.retriever, prompt)
                
                # Create QA chain
                qa_chain = create_custom_qa_chain(st.session_state.retriever, prompt_type)
                
                # Get response
                response = qa_chain.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Format response
                if prompt_type == "structured":
                    # Try to extract structured JSON
                    json_data = extract_json_from_response(response_text)
                    if json_data:
                        formatted_response = format_response(json.dumps(json_data))
                    else:
                        formatted_response = response_text
                else:
                    formatted_response = response_text
                
                # Display response
                response_placeholder.markdown(formatted_response)
                
                # Add response to chat history
                st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                
                # Display sources
                with sources_container:
                    st.markdown("#### Sources")
                    sources = extract_sources_from_docs(docs)
                    
                    for i, source in enumerate(sources):
                        st.markdown(f"""
                        **Source {i+1}:** {source.get('source', 'Unknown')} (Page {source.get('page', 'Unknown')})  
                        *Preview:* {source.get('preview', 'No preview available')}
                        """)

# Display a welcome message when no messages exist
if not st.session_state.messages:
    st.info("👋 Welcome! Upload diabetes-related PDF documents and ask questions to get started.")

# Run the Streamlit app
if __name__ == "__main__":
    # This is handled by Streamlit's execution model
    pass