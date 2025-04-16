from typing import List
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import GOOGLE_API_KEY, GEMINI_EMBEDDING_MODEL

def initialize_gemini_embeddings():
    """
    Initialize Gemini embeddings model.
    
    Returns:
        Configured embeddings model
    """
    # Configure Google Gemini API
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        task_type="retrieval_document"
    )
    
    return embeddings

def get_text_embedding(text: str) -> List[float]:
    """
    Get embedding for a single text.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector
    """
    embeddings = initialize_gemini_embeddings()
    return embeddings.embed_query(text)

def batch_embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batch embed multiple texts.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    embeddings = initialize_gemini_embeddings()
    return [embeddings.embed_query(text) for text in texts]