import os
import json
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
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = save_uploaded_file(uploaded_file)
                    file_paths.append(file_path)
                    st.session_state.uploaded_files.append({
                        "name": uploaded_file.name,
                        "path": file_path
                    })
                
                documents = load_pdf_documents(file_paths)
                
                if documents:
                    document_chunks = split_documents(documents)
                    enhanced_chunks = enhance_documents(document_chunks)
                    
                    vectorstore = create_vectorstore(enhanced_chunks)
                    save_vectorstore(vectorstore)
                    
                    retriever = create_retriever(vectorstore)
                    st.session_state.retriever = retriever
                    st.session_state.vectorstore_ready = True
                    
                    st.success(f"Successfully processed {len(documents)} documents ({len(enhanced_chunks)} chunks)")
                else:
                    st.error("No documents were successfully loaded. Please check the files and try again.")
    
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

    st.markdown("<hr/>", unsafe_allow_html=True)

    if st.session_state.uploaded_files:
        st.header("📄 Uploaded Documents")
        for file in st.session_state.uploaded_files:
            st.write(f"- {file['name']}")

    st.markdown("<hr/>", unsafe_allow_html=True)

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

# Display chat history
st.subheader("💬 Chat History")
for i, message in enumerate(st.session_state.messages):
    role = "👤 You" if message["role"] == "user" else "🤖 Assistant"
    st.markdown(f"**{role}:** {message['content']}")

# Input box for chat prompt (compatible with older Streamlit)
user_input = st.text_input("Ask a question about diabetes management:", key="user_input")
submit_button = st.button("Send")

if submit_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"**👤 You:** {user_input}")

    if not st.session_state.vectorstore_ready:
        st.warning("Please upload and process documents before asking questions.")
    else:
        with st.spinner("Thinking..."):
            docs = retrieve_documents(st.session_state.retriever, user_input)
            qa_chain = create_custom_qa_chain(st.session_state.retriever, prompt_type)
            response = qa_chain.invoke(user_input)
            response_text = response.content if hasattr(response, 'content') else str(response)

            if prompt_type == "structured":
                json_data = extract_json_from_response(response_text)
                if json_data:
                    formatted_response = format_response(json.dumps(json_data))
                else:
                    formatted_response = response_text
            else:
                formatted_response = response_text

            st.markdown(f"**🤖 Assistant:** {formatted_response}")
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})

            st.markdown("#### Sources")
            sources = extract_sources_from_docs(docs)
            for i, source in enumerate(sources):
                st.markdown(f"""
                **Source {i+1}:** {source.get('source', 'Unknown')} (Page {source.get('page', 'Unknown')})  
                *Preview:* {source.get('preview', 'No preview available')}
                """)

if not st.session_state.messages:
    st.info("👋 Welcome! Upload diabetes-related PDF documents and ask questions to get started.")


# Run the Streamlit app
if __name__ == "__main__":
    # This is handled by Streamlit's execution model
    pass