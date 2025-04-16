import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Vector store settings
VECTOR_STORE_PATH = "vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Gemini model settings
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
GEMINI_GENERATION_MODEL = "gemini-1.5-pro"

# RAG settings
TOP_K_RESULTS = 5

# Application settings
APP_TITLE = "Diabetes Management Assistant"
APP_DESCRIPTION = "A RAG-based Medical GenAI chatbot for diabetic patients"

# Few-shot examples for the chatbot
FEW_SHOT_EXAMPLES = [
    {
        "question": "What are the symptoms of hypoglycemia?",
        "answer": "Symptoms of hypoglycemia (low blood sugar) typically include shakiness, dizziness, sweating, hunger, irritability, confusion, rapid heartbeat, and in severe cases, loss of consciousness. It's important to recognize these symptoms early and treat hypoglycemia promptly with fast-acting carbohydrates."
    },
    {
        "question": "How often should I check my blood sugar?",
        "answer": "The frequency of blood sugar monitoring depends on your specific diabetes management plan, type of diabetes, and medications. Generally, people with type 1 diabetes may need to check 4-10 times daily, while those with type 2 might check 1-4 times daily. Consult with your healthcare provider for personalized recommendations based on your specific situation."
    },
    {
        "question": "What is a good HbA1c target?",
        "answer": "For most adults with diabetes, the American Diabetes Association recommends an HbA1c target of less than 7%. However, targets may be personalized based on age, duration of diabetes, comorbid conditions, and individual factors. Your healthcare provider can help determine the most appropriate target for your specific situation."
    }
]

# Structured output format for health recommendations
STRUCTURED_OUTPUT_FORMAT = {
    "answer": "Direct response to the user's question",
    "medical_context": "Relevant medical information from the documents",
    "recommendations": "Practical suggestions or advice",
    "follow_up": "Important points to discuss with healthcare provider"
}