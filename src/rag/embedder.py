from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

# Using a local embedding model — no API calls, no rate limits.
# all-MiniLM-L6-v2: ~80 MB, fast, good quality, 384-dim vectors.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embeddings_model():
    """Returns a local HuggingFace sentence-transformers embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_faiss_index(documents: list[Document]) -> FAISS:
    """
    Takes a list of LangChain Documents and embeds them using a local
    sentence-transformers model, then stores the vectors in FAISS.
    """
    embeddings = get_embeddings_model()
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
    return FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=allow_dangerous_deserialization)

def merge_faiss_indices(stores: list[FAISS]) -> FAISS:
    """
    Merges a list of FAISS vector stores into a single unified store.
    The first store is used as the base, and the rest are merged into it.
    """
    if not stores:
        raise ValueError("No FAISS stores provided to merge.")
    
    base = stores[0]
    for store in stores[1:]:
        base.merge_from(store)
    
    return base
