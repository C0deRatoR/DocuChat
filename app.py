import streamlit as st
import os
from dotenv import load_dotenv

from src.utils.pdf_utils import PDFProcessingError
from src.rag.ingest import process_pdf
from src.rag.embedder import build_faiss_index, save_faiss_index, load_faiss_index
from src.rag.chain import build_qa_chain

# Load environment variables
load_dotenv()

st.set_page_config(page_title="DocuChat", page_icon="📄", layout="centered")

st.title("📄 DocuChat")
st.markdown("Upload a PDF document and ask questions about its content. Powered by **Gemini 1.5 Flash** and **FAISS**.")

# Initialize session state for chat history and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar for controls
with st.sidebar:
    st.header("Document Setup")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing PDF and generating embeddings..."):
                try:
                    # 1. Read file bytes
                    file_bytes = uploaded_file.read()
                    
                    # 2. Process to LangChain documents
                    docs = process_pdf(file_bytes)
                    
                    # 3. Build FAISS index
                    vector_store = build_faiss_index(docs)
                    
                    # 4. Save to session state
                    st.session_state.vector_store = vector_store
                    save_faiss_index(vector_store)
                    
                    # Reset chat history on new document
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    
                    st.success("Document processed successfully!")
                except PDFProcessingError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    
    elif os.path.exists("/tmp/faiss_index") and st.session_state.vector_store is None:
        # Load existing index if present
        try:
            st.session_state.vector_store = load_faiss_index()
            st.success("Loaded previous document index.")
        except:
            pass

# Main Chat Interface
if st.session_state.vector_store is None:
    st.info("👈 Please upload and process a PDF document to begin chatting.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📄 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Page {source['page']}**: {source['content']}...")
                        if i < len(message["sources"]) - 1:
                            st.divider()

    # User input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build chain and generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                qa_chain = build_qa_chain(st.session_state.vector_store)
                
                try:
                    response = qa_chain.invoke({
                        "question": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    answer = response["answer"]
                    source_docs = response.get("source_documents", [])
                    
                    # Format sources
                    sources = []
                    for doc in source_docs:
                        page_num = doc.metadata.get("page_number", "Unknown")
                        # Preview string ~200 chars
                        preview = doc.page_content.replace('\n', ' ')[:200]
                        sources.append({"page": page_num, "content": preview})
                        
                    # Display Answer
                    st.markdown(answer)
                    
                    # Display Sources if any
                    if sources:
                        with st.expander("📄 Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Page {source['page']}**: {source['content']}...")
                                if i < len(sources) - 1:
                                    st.divider()
                    
                    # Update session state with assistant's response
                    ass_msg = {"role": "assistant", "content": answer}
                    if sources:
                        ass_msg["sources"] = sources
                        
                    st.session_state.messages.append(ass_msg)
                    st.session_state.chat_history.append((prompt, answer))
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
