import os
from typing import List, Dict, Any
import faiss
from langchain_community.vectorstores import FAISS

from src.embeddings.gemini_embeddings import initialize_gemini_embeddings
from src.config import VECTOR_STORE_PATH

def create_vectorstore(documents: List) -> FAISS:
    """
    Create a FAISS vector store from documents.
    
    Args:
        documents: List of document objects
        
    Returns:
        FAISS vector store
    """
    embeddings = initialize_gemini_embeddings()
    
    # Create vector store
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    return vectorstore

def save_vectorstore(vectorstore: FAISS, directory: str = VECTOR_STORE_PATH) -> None:
    """
    Save the vector store to disk.
    
    Args:
        vectorstore: FAISS vector store
        directory: Directory to save the vector store
    """
    os.makedirs(directory, exist_ok=True)
    vectorstore.save_local(directory)
    print(f"Vector store saved to {directory}")

def load_vectorstore(directory: str = VECTOR_STORE_PATH) -> FAISS:
    """
    Load a vector store from disk.
    
    Args:
        directory: Directory containing the vector store
        
    Returns:
        FAISS vector store
    """
    embeddings = initialize_gemini_embeddings()
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Vector store directory {directory} not found")
    
    vectorstore = FAISS.load_local(directory, embeddings)
    print(f"Vector store loaded from {directory}")
    
    return vectorstore

def add_documents_to_vectorstore(vectorstore: FAISS, documents: List) -> FAISS:
    """
    Add documents to an existing vector store.
    
    Args:
        vectorstore: Existing FAISS vector store
        documents: New documents to add
        
    Returns:
        Updated FAISS vector store
    """
    vectorstore.add_documents(documents)
    print(f"Added {len(documents)} documents to the vector store")
    
    return vectorstore