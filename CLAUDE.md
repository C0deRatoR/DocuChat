# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Docker
docker-compose up --build  # Access at http://localhost:8501
```

## Architecture Overview

**DocuChat** is a RAG-powered PDF Q&A application with a modular `src/` package:

```
src/
├── rag/           # RAG pipeline components
│   ├── ingest.py      # PDF → text extraction + chunking (RecursiveCharacterTextSplitter)
│   ├── embedder.py    # HuggingFace embeddings (all-MiniLM-L6-v2) + FAISS index management
│   ├── retriever.py   # FAISS retriever wrapper (k=5 default)
│   ├── chain.py       # Groq LLM (Llama-3.3-70B) + conversational QA chain
│   └── summarizer.py  # Document summary generation
└── utils/
    └── pdf_utils.py   # PyMuPDF validation & page extraction
```

**Data Flow:**
1. **Ingest**: PDF validated → text extracted with page metadata → split into chunks
2. **Embed**: Chunks embedded via local sentence-transformers → FAISS vector store
3. **Retrieve**: Similarity search returns top-k chunks
4. **Generate**: Groq LLM produces grounded answers with source citations

**Key Design Decisions:**
- Uses **Groq** for LLM (free tier, 14.4K req/day limit) instead of Gemini for cost
- Uses **local HuggingFace embeddings** (all-MiniLM-L6-v2) to avoid embedding API costs
- FAISS index cached to `/tmp/faiss_index` for session persistence
- Multi-document support: per-file indices merged into unified vector store
- Retry logic (exponential backoff) for transient API errors

## Common Operations

```bash
# Process a single PDF (programmatic)
from src.rag.ingest import process_pdf
from src.rag.embedder import build_faiss_index

docs = process_pdf(pdf_bytes, source_filename="myfile.pdf")
vector_store = build_faiss_index(docs)

# Build QA chain and query
from src.rag.chain import build_qa_chain
qa_chain = build_qa_chain(vector_store)
response = qa_chain.invoke({"question": "Your question", "chat_history": []})
```

## Environment Variables

Required in `.env`:
- `GEMINI_API_KEY` — Currently unused (legacy from earlier Gemini implementation)
- Groq API key should be set as `GROQ_API_KEY` (not yet enforced in code)

## Known Limitations

- No OCR support (image-only/scanned PDFs fail)
- No password-protected PDF support
- No chat history persistence across sessions (in-memory only)
- No multi-user authentication
