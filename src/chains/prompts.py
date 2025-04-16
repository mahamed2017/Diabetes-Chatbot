from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

from src.config import FEW_SHOT_EXAMPLES, STRUCTURED_OUTPUT_FORMAT

# System prompt for the QA chain
SYSTEM_PROMPT = """You are DiabetesAssistant, a medical AI specialized in diabetes management and patient education.
Your job is to provide accurate, helpful information about diabetes based on verified medical literature.

Follow these guidelines:
1. Provide detailed, accurate medical information based on the context provided.
2. Be empathetic and considerate of patient concerns.
3. Focus on evidence-based medical advice.
4. Acknowledge the limits of your knowledge when appropriate.
5. Emphasize the importance of consulting healthcare providers for personal medical advice.
6. Never make up information that isn't supported by the context.
7. Maintain a professional yet approachable tone.

When answering questions:
- First analyze the question to understand what the patient is asking
- Review the provided context carefully
- Formulate a comprehensive, accurate response
- Include references to specific sections of the documents when relevant
- For questions outside your knowledge or context, acknowledge limitations
"""

# Base QA prompt template
qa_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        """I need information about diabetes based on medical literature.
        
Context information from medical documents:
{context}

Patient question: {question}

Please provide a helpful, accurate response."""
    )
])

# Few-shot learning prompt template
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}"
)

few_shot_prompt_template = FewShotPromptTemplate(
    examples=FEW_SHOT_EXAMPLES,
    example_prompt=example_prompt,
    prefix="""You are a diabetes management AI assistant. Here are some examples of questions and high-quality answers:""",
    suffix="""Question: {question}\nContext: {context}\n\nAnswer:""",
    input_variables=["question", "context"]
)

# Structured output prompt template
structured_output_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        """I need information about diabetes based on medical literature.
        
Context information from medical documents:
{context}

Patient question: {question}

Please provide a structured response with the following sections:
1. Answer: Direct response to the question
2. Medical Context: Relevant information from the documents
3. Recommendations: Practical advice based on medical guidelines
4. Follow-up: Important points to discuss with healthcare providers

Format your response as JSON following this structure:
{structured_format}"""
    )
])

def get_qa_prompt():
    """Return the base QA prompt template"""
    return qa_prompt_template

def get_few_shot_prompt():
    """Return the few-shot prompt template"""
    return few_shot_prompt_template

def get_structured_output_prompt():
    """Return the structured output prompt template"""
    return structured_output_prompt.partial(structured_format=str(STRUCTURED_OUTPUT_FORMAT))