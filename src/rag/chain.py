from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from src.rag.retriever import get_retriever

def get_qa_prompt() -> PromptTemplate:
    template = """You are a precise document assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find this in the documents."
Do not use any external knowledge.
When the context comes from multiple documents, reference the source document name in your answer where relevant.

Context:
{context}

Question: {question}

Answer:"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def build_qa_chain(vector_store: FAISS):
    """
    Builds the Conversational Retrieval Chain.
    Uses Gemini 1.5 Flash for answering and a custom strict grounding prompt.
    Returns the chain.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retriever = get_retriever(vector_store, k=5)
    
    qa_prompt = get_qa_prompt()
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    
    return chain
