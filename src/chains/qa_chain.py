from typing import Dict, Any, List
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

from src.chains.prompts import get_qa_prompt, get_few_shot_prompt, get_structured_output_prompt
from src.config import GOOGLE_API_KEY, GEMINI_GENERATION_MODEL

def initialize_llm():
    """
    Initialize the Gemini language model.
    
    Returns:
        Configured Gemini LLM
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_GENERATION_MODEL,
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True
    )
    
    return llm

def format_docs(docs: List) -> str:
    """
    Format retrieved documents into a single context string.
    
    Args:
        docs: List of retrieved documents
        
    Returns:
        Formatted context string
    """
    return "\n\n".join([doc.page_content for doc in docs])

def create_qa_chain(retriever):
    """
    Create a standard question-answering chain.
    
    Args:
        retriever: Document retriever
        
    Returns:
        QA chain
    """
    llm = initialize_llm()
    
    # Create a memory buffer to store conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        chain_type="stuff",
        get_chat_history=lambda h: h,
        verbose=True
    )
    
    return qa_chain

def create_custom_qa_chain(retriever, prompt_type="standard"):
    """
    Create a custom QA chain with specified prompt type.
    
    Args:
        retriever: Document retriever
        prompt_type: Type of prompt to use (standard, few_shot, structured)
        
    Returns:
        Custom QA chain
    """
    llm = initialize_llm()
    
    # Select prompt template based on type
    if prompt_type == "few_shot":
        prompt = get_few_shot_prompt()
    elif prompt_type == "structured":
        prompt = get_structured_output_prompt()
    else:  # standard
        prompt = get_qa_prompt()
    
    # Create the custom QA chain
    qa_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return qa_chain

def extract_sources_from_docs(docs):
    """
    Extract source information from retrieved documents.
    
    Args:
        docs: List of retrieved documents
        
    Returns:
        List of source information dictionaries
    """
    sources = []
    
    for doc in docs:
        source = {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown"),
            "preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        }
        sources.append(source)
    
    return sources