import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """
    Clean and normalize document text.
    
    Args:
        text: Raw text from document
        
    Returns:
        Cleaned text
    """
    # Remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers if detectable
    text = re.sub(r'\b(?:Page|page)\s+\d+\s+(?:of|/)\s+\d+\b', '', text)
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n')
    
    return text.strip()

def extract_metadata(document) -> Dict[str, Any]:
    """
    Extract and enhance document metadata.
    
    Args:
        document: Document object
        
    Returns:
        Enhanced metadata dictionary
    """
    metadata = document.metadata.copy()
    
    # Add document section detection if possible
    text = document.page_content
    
    # Try to identify document sections (headers)
    headers = re.findall(r'^(#+\s+.+)$|^([A-Z][A-Z\s]+):$', text, re.MULTILINE)
    if headers:
        flattened_headers = [h[0] or h[1] for h in headers if h[0] or h[1]]
        metadata['detected_headers'] = flattened_headers
    
    return metadata

def enhance_documents(documents: List) -> List:
    """
    Clean text and enhance metadata for all documents.
    
    Args:
        documents: List of document objects
        
    Returns:
        List of enhanced document objects
    """
    enhanced_docs = []
    
    for doc in documents:
        # Clean the text
        cleaned_text = clean_text(doc.page_content)
        
        # Extract and enhance metadata
        enhanced_metadata = extract_metadata(doc)
        
        # Create new document with cleaned text and enhanced metadata
        doc.page_content = cleaned_text
        doc.metadata = enhanced_metadata
        enhanced_docs.append(doc)
    
    return enhanced_docs