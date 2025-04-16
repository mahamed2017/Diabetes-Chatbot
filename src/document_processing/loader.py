import os
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load PDF documents from the given file paths.
    
    Args:
        file_paths: List of paths to PDF files
        
    Returns:
        List of documents with text content and metadata
    """
    documents = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        try:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"Successfully loaded document: {file_path}")
        except Exception as e:
            print(f"Error loading document {file_path}: {str(e)}")
    
    return documents

def split_documents(documents):
    """
    Split documents into smaller chunks for better processing.
    
    Args:
        documents: List of document objects
        
    Returns:
        List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    document_chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(document_chunks)} chunks")
    
    return document_chunks