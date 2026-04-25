import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv

from src.utils.pdf_utils import PDFProcessingError
from src.rag.ingest import process_pdf
from src.rag.embedder import build_faiss_index, save_faiss_index, load_faiss_index, merge_faiss_indices
from src.rag.summarizer import summarize_documents
from src.rag.chain import build_qa_chain

# Load environment variables
load_dotenv()

st.set_page_config(page_title="DocuChat", page_icon="📄", layout="centered")

st.title("📄 DocuChat")
st.markdown("Upload one or more PDF documents and ask questions about their content. Powered by **Gemini 2.5 Flash** and **FAISS**.")

# Initialize session state for chat history and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = None
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []

# Sidebar for controls
with st.sidebar:
    st.header("Document Setup")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )
    
    if uploaded_files:
        # Show currently selected files
        st.caption(f"**{len(uploaded_files)}** file(s) selected")
        for f in uploaded_files:
            st.markdown(f"- `{f.name}`")

        if st.button("Process Documents"):
            all_docs = []           # Merged list of LangChain Documents
            per_file_docs = {}      # filename -> list[Document] for per-doc summaries
            faiss_stores = []       # One FAISS store per file, merged later
            errors = []

            progress_bar = st.progress(0, text="Starting…")

            for idx, uploaded_file in enumerate(uploaded_files):
                file_label = uploaded_file.name
                progress_bar.progress(
                    (idx) / len(uploaded_files),
                    text=f"Processing **{file_label}**…",
                )
                try:
                    file_bytes = uploaded_file.read()
                    docs = process_pdf(file_bytes, source_filename=file_label)
                    all_docs.extend(docs)
                    per_file_docs[file_label] = docs

                    # Build a per-file FAISS index
                    store = build_faiss_index(docs)
                    faiss_stores.append(store)

                except PDFProcessingError as e:
                    errors.append(f"**{file_label}**: {e}")
                except Exception as e:
                    errors.append(f"**{file_label}**: Unexpected error — {e}")

            # --- Merge indices ---
            if faiss_stores:
                progress_bar.progress(0.9, text="Merging indices…")
                merged_store = merge_faiss_indices(faiss_stores)
                st.session_state.vector_store = merged_store
                save_faiss_index(merged_store)

                # Track which files are loaded
                st.session_state.uploaded_filenames = list(per_file_docs.keys())

                # Reset chat history on new document set
                st.session_state.messages = []
                st.session_state.chat_history = []

                progress_bar.progress(1.0, text="Done!")
                st.success(
                    f"✅ Processed **{len(faiss_stores)}** document(s) — "
                    f"**{len(all_docs)}** chunks indexed."
                )
            else:
                progress_bar.empty()
                st.error("No documents could be processed.")

            # Show per-file errors
            for err in errors:
                st.error(err)

            # --- Generate per-document summaries ---
            if st.session_state.vector_store is not None and per_file_docs:
                with st.spinner("Generating document summaries…"):
                    summaries = {}
                    for fname, docs in per_file_docs.items():
                        try:
                            summaries[fname] = summarize_documents(docs)
                        except Exception as e:
                            summaries[fname] = f"_Could not generate summary: {e}_"
                    st.session_state.doc_summary = summaries

    elif os.path.exists("/tmp/faiss_index") and st.session_state.vector_store is None:
        # Load existing index if present
        try:
            st.session_state.vector_store = load_faiss_index()
            st.success("Loaded previous document index.")
        except:
            pass

    # --- Loaded documents panel ---
    if st.session_state.uploaded_filenames:
        st.divider()
        st.header("Loaded Documents")
        for fname in st.session_state.uploaded_filenames:
            st.markdown(f"📎 `{fname}`")

    # --- Export Chat History ---
    if st.session_state.messages:
        st.divider()
        st.header("Export")

        def _build_chat_markdown() -> str:
            """Formats the current chat history as a downloadable Markdown string."""
            lines = [
                "# DocuChat – Conversation Export",
                f"_Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
                "",
                "---",
                "",
            ]
            for msg in st.session_state.messages:
                role_label = "🧑 **You**" if msg["role"] == "user" else "🤖 **DocuChat**"
                lines.append(f"### {role_label}")
                lines.append("")
                lines.append(msg["content"])
                lines.append("")
                if "sources" in msg:
                    lines.append("<details><summary>📄 Sources</summary>")
                    lines.append("")
                    for src in msg["sources"]:
                        lines.append(f"- **{src['source']}** · Page {src['page']}: {src['content']}…")
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")
                lines.append("---")
                lines.append("")
            return "\n".join(lines)

        st.download_button(
            label="📥 Download Chat as Markdown",
            data=_build_chat_markdown(),
            file_name=f"docuchat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

# Main Chat Interface
if st.session_state.vector_store is None:
    st.info("👈 Please upload and process one or more PDF documents to begin chatting.")
else:
    # --- Document Summaries ---
    if st.session_state.doc_summary:
        summaries = st.session_state.doc_summary
        if isinstance(summaries, dict):
            # Multi-document summaries
            with st.expander(f"📋 Document Summaries ({len(summaries)} documents)", expanded=False):
                for fname, summary in summaries.items():
                    st.markdown(f"### 📎 {fname}")
                    st.markdown(summary)
                    st.divider()
        else:
            # Legacy single-document summary (backwards compat)
            with st.expander("📋 Document Summary", expanded=True):
                st.markdown(summaries)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📄 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**{source['source']}** · Page {source['page']}: {source['content']}...")
                        if i < len(message["sources"]) - 1:
                            st.divider()

    # User input
    if prompt := st.chat_input("Ask a question about your documents..."):
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
                    
                    # Format sources — now includes source filename
                    sources = []
                    for doc in source_docs:
                        page_num = doc.metadata.get("page_number", "Unknown")
                        source_file = doc.metadata.get("source", "Unknown")
                        # Preview string ~200 chars
                        preview = doc.page_content.replace('\n', ' ')[:200]
                        sources.append({
                            "page": page_num,
                            "source": source_file,
                            "content": preview,
                        })
                        
                    # Display Answer
                    st.markdown(answer)
                    
                    # Display Sources if any
                    if sources:
                        with st.expander("📄 Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**{source['source']}** · Page {source['page']}: {source['content']}...")
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
