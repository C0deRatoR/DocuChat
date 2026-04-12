from langchain_community.vectorstores import FAISS

def get_retriever(vector_store: FAISS, k: int = 5):
    """
    Returns a retriever initialized from the FAISS vector store.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
