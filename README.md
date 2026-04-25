# DocuChat — RAG-Powered PDF Q&A

DocuChat is a Retrieval-Augmented Generation (RAG) powered web application built to allow natural language conversations with PDF documents. Answers are strictly grounded in the uploaded document and include direct page citations, eliminating hallucination.

## 🚀 Features
- **Multi-PDF Processing**: Upload multiple documents up to 20MB. Automatically extracts text using PyMuPDF and merges them into a single knowledge base.
- **RAG Pipeline**: Splits text into context-preserving chunks, embedded locally using `all-MiniLM-L6-v2` via sentence-transformers (zero API cost/rate limits).
- **In-Memory Vector Search**: Uses FAISS for ultra-fast, local similarity search.
- **Smart Grounding**: Powered by Ollama's `llama3.2` model. Capable of synthesizing answers across multiple documents locally while providing accurate page and file citations.
- **Interactive UI**: Clean, responsive Streamlit chat interface with full conversation history and per-document summaries.

## 🏗 Stack Overview
- **Language**: Python 3.11
- **LLM**: Ollama (`llama3.2` 3B) — free, runs locally
- **Embeddings**: Local HuggingFace sentence-transformers (`all-MiniLM-L6-v2`)
- **Orchestration**: LangChain
- **Vector DB**: FAISS (CPU)
- **Frontend**: Streamlit
- **Containerization**: Docker & Docker Compose

## 🛠️ Setup Instructions (Local)
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/docuchat.git
   cd docuchat
   ```
2. **Setup virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Install & Start Ollama**
   ```bash
   # Install Ollama: https://ollama.ai
   ollama pull llama3.2
   ollama serve
   ```
5. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 🐳 Docker Setup
**Note:** Ollama requires a running server. For Docker, either:
- Run Ollama on your host and expose port 11434, or
- Use the host network mode

1. Start Ollama: `ollama serve` (on host or in a separate container)
2. Build and start the container:
   ```bash
   docker-compose up --build
   ```
3. Access the app locally at: `http://localhost:8501`.

## 🧠 Architecture
1. **Ingest**: PDF parsed via PyMuPDF. Text heavily chunked using LangChain's `RecursiveCharacterTextSplitter`.
2. **Embed**: Text chunks mapped to dense 384-dimensional vectors by the local `all-MiniLM-L6-v2` model.
3. **Index**: Embeddings loaded into FAISS vector store. Temporary cached copies store to `/tmp/faiss_index` for persistence over Streamlit session reruns.
4. **Query**: User queries embedded -> Similarity Search via FAISS bounds top-k contextual text blocks.
5. **Generate**: Extracted chunk bodies injected into strict LangChain prompt alongside the question.
6. **Cite**: Page-level metadata preserved at chunking is surfaced alongside the generated answers.

## ⚠️ Notes
- Does not currently support scanned (image-only) PDFs or password-protected files.
- Max PDF limit optimized natively for <20MB for fast pipeline throughput.

---
**Author**: SY BTech CS student
