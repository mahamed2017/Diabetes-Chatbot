# 🩺 Diabetes RAG Assistant

A medical GenAI chatbot designed to assist with diabetes-related questions using your uploaded PDF documents. It leverages Retrieval-Augmented Generation (RAG), Google Gemini API, FAISS, and LangChain, all wrapped in a friendly Streamlit UI.

---

## 🚀 Features

- 🔍 **PDF-based Q&A**: Ask questions based on uploaded diabetes-related documents.
- 🧠 **RAG-powered**: Combines vector search with Google Gemini LLM responses.
- 📄 **Multi-doc support**: Upload and query multiple PDFs at once.
- ✨ **Prompt styles**: Choose from Standard, Few-shot, or Structured output formats.
- ⚡ **Fast local search**: Powered by FAISS vector database.
- 🖥️ **Web interface**: Easy-to-use interface built with Streamlit.

---

## 🗂️ Project Structure

diabetes_rag_assistant/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (API keys)
├── .gitignore              # Git ignore file
├── README.md               # Project documentation
├── data/                   # Directory for PDF documents
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── document_processing/
│   │   ├── __init__.py
│   │   ├── loader.py       # Document loading utilities
│   │   └── processor.py    # Text processing utilities
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── gemini_embeddings.py  # Gemini embedding utilities
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vectorstore.py  # FAISS vector store management
│   │   └── retriever.py    # Document retrieval utilities
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── qa_chain.py     # Question-answering chain
│   │   └── prompts.py      # Custom prompt templates
│   └── utils/
│       ├── __init__.py
│       └── helpers.py      # Helper functions
└── tests/                  # Test cases
    └── __init__.py

# ⚙️ Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/diabetes_rag_assistant.git
cd diabetes_rag_assistant

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key

GEMINI_API_KEY=your_google_gemini_api_key

# Run the app

streamlit run app.py

# 🧠 How to Use
Upload one or more diabetes-related PDF files using the left sidebar.

Choose your preferred prompt style:

Standard

Few-shot

Structured (JSON format)

Enter your question in the input box.

The assistant will fetch the most relevant context and generate an answer.


# 📄 License
This project is licensed under the MIT License.

# Contact
E-mail:mahamed.02@outlook.com








