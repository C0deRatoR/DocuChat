from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from src.rag.retriever import get_retriever

def get_qa_prompt() -> PromptTemplate:
    template = """You are a helpful and intelligent document assistant. Your job is to answer user questions based on the provided context extracted from their uploaded documents.

Instructions:
- Base your answer ONLY on the context provided below — do NOT use external knowledge.
- You CAN and SHOULD synthesize, summarize, and infer from the context. For example, if the context contains education details, work experience, and projects, you can infer that the document is a resume/CV.
- When the context comes from multiple documents, mention which document the information is from.
- If the context genuinely does not contain ANY information relevant to the question, say "I could not find relevant information about this in the uploaded documents."
- Be concise but thorough. Use markdown formatting (bold, bullet points, etc.) for readability.

Context:
{context}

Question: {question}

Answer:"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def get_condense_prompt() -> PromptTemplate:
    """Prompt for condensing a follow-up question with chat history into a standalone question."""
    template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that captures the full intent.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""
    return PromptTemplate(
        template=template,
        input_variables=["chat_history", "question"]
    )

def build_qa_chain(vector_store: FAISS):
    """
    Builds the Conversational Retrieval Chain.
    Uses Groq (Llama 3.3 70B) for answering — free tier with 14,400 req/day.
    Returns the chain.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    retriever = get_retriever(vector_store, k=5)
    
    qa_prompt = get_qa_prompt()
    condense_prompt = get_condense_prompt()
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_prompt,
    )
    
    return chain
