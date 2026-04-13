from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

def get_embeddings_model():
    """Returns the Gemini embedding model configured via LangChain."""
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def build_faiss_index(documents: list[Document]) -> FAISS:
    """
    Takes a list of LangChain Documents and embeds them using Gemini's gemini-embedding-001 model,
    then stores the resulting vectors in a LangChain FAISS vector store.
    """
    embeddings = get_embeddings_model()
    
    # FAISS will handle chunking these batches to Google's API automatically
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store

def save_faiss_index(vector_store: FAISS, folder_path: str = "/tmp/faiss_index"):
    """Saves the FAISS index to disk."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    vector_store.save_local(folder_path)

def load_faiss_index(folder_path: str = "/tmp/faiss_index", allow_dangerous_deserialization: bool = True) -> FAISS:
    """Loads a previously saved FAISS index from disk."""
    embeddings = get_embeddings_model()
    # allow_dangerous_deserialization is needed in newer versions of LangChain for local FAISS loads
    return FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=allow_dangerous_deserialization)
