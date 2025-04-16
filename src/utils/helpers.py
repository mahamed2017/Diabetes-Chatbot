import os
import json
import tempfile
from datetime import datetime
from typing import List, Dict, Any

def format_response(response: str) -> str:
    """
    Format the response for better readability.
    
    Args:
        response: Raw response from LLM
        
    Returns:
        Formatted response
    """
    # Try to detect if the response is in JSON format
    try:
        # If it's JSON formatted, parse and reformat it
        data = json.loads(response)
        return format_structured_output(data)
    except json.JSONDecodeError:
        # If not JSON, return the original response
        return response

def format_structured_output(data: Dict[str, Any]) -> str:
    """
    Format structured output for display.
    
    Args:
        data: Structured output data
        
    Returns:
        Formatted output string
    """
    formatted_response = ""
    
    if "answer" in data:
        formatted_response += f"### Answer\n{data['answer']}\n\n"
    
    if "medical_context" in data:
        formatted_response += f"### Medical Context\n{data['medical_context']}\n\n"
    
    if "recommendations" in data:
        formatted_response += f"### Recommendations\n{data['recommendations']}\n\n"
    
    if "follow_up" in data:
        formatted_response += f"### Important Follow-up Points\n{data['follow_up']}\n\n"
    
    return formatted_response

def save_uploaded_file(uploaded_file) -> str:
    """
    Save an uploaded file to a temporary location.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Path to the saved file
    """
    # Create a temporary directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), "diabetes_assistant")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a filename with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(temp_dir, f"{timestamp}_{uploaded_file.name}")
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response if present.
    
    Args:
        response: LLM response string
        
    Returns:
        Extracted JSON data or empty dict
    """
    # Try to find JSON pattern in the response
    json_pattern = r'```json\s*([\s\S]*?)\s*```|{\s*"[^"]+"\s*:|{\s*\'[^\']+\'\s*:'
    import re
    
    match = re.search(json_pattern, response)
    if match:
        json_str = match.group(1) if match.group(1) else response
        
        # Clean the string if needed
        json_str = json_str.strip()
        if not (json_str.startswith('{') and json_str.endswith('}')):
            # Find the JSON object boundaries
            start = json_str.find('{')
            end = json_str.rfind('}')
            if start != -1 and end != -1:
                json_str = json_str[start:end+1]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}
    
    return {}

def get_chat_history(messages: List[Dict[str, str]]) -> str:
    """
    Format chat history for the LLM context.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Formatted chat history string
    """
    chat_history = ""
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "user":
            chat_history += f"User: {content}\n"
        elif role == "assistant":
            chat_history += f"Assistant: {content}\n"
    
    return chat_history