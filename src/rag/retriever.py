from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS

from src.config import TOP_K_RESULTS

def create_retriever(vectorstore: FAISS):
    """
    Create a document retriever from a vector store.
    
    Args:
        vectorstore: FAISS vector store
        
    Returns:
        Document retriever
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )
    
    return retriever

def retrieve_documents(retriever, query: str) -> List:
    """
    Retrieve relevant documents for a query.
    
    Args:
        retriever: Document retriever
        query: User query
        
    Returns:
        List of relevant documents
    """
    documents = retriever.get_relevant_documents(query)
    return documents

def get_document_sources(documents: List) -> List[Dict[str, Any]]:
    """
    Extract source information from documents.
    
    Args:
        documents: List of retrieved documents
        
    Returns:
        List of document sources with metadata
    """
    sources = []
    
    for i, doc in enumerate(documents):
        source_info = {
            "content_preview": doc.page_content[:100] + "...",
            "source": doc.metadata.get("source", f"Document {i+1}"),
            "page": doc.metadata.get("page", "Unknown")
        }
        sources.append(source_info)
    
    return sources

def create_mmr_retriever(vectorstore: FAISS):
    """
    Create a Maximum Marginal Relevance retriever for diversity in results.
    
    Args:
        vectorstore: FAISS vector store
        
    Returns:
        MMR retriever
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K_RESULTS,
            "fetch_k": TOP_K_RESULTS * 2,
            "lambda_mult": 0.7  # Controls diversity (0 = max diversity, 1 = max relevance)
        }
    )
    
    return retriever