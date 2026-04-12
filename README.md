# DocuChat — RAG-Powered PDF Q&A

DocuChat is a Retrieval-Augmented Generation (RAG) powered web application built to allow natural language conversations with PDF documents. Answers are strictly grounded in the uploaded document and include direct page citations, eliminating hallucination.

## 🚀 Features
- **PDF Processing**: Upload documents up to 20MB. Automatically extracts text using PyMuPDF.
- **RAG Pipeline**: Splits text into context-preserving chunks, embedded using Gemini `embedding-001`.
- **In-Memory Vector Search**: Uses FAISS for ultra-fast, local similarity search.
- **Strict Grounding**: Gemini 1.5 Flash generates answers **only** from the provided text, adding source page citations to every response.
- **Interactive UI**: Clean, responsive Streamlit chat interface with full conversation history during the session.

## 🏗 Stack Overview
- **Language**: Python 3.11
- **LLM/Embeddings**: Google Gemini API (`1.5-Flash`, `embedding-001`)
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
4. **Configure Environment variables**
   Copy the example environment file and add your `GEMINI_API_KEY`:
   ```bash
   cp .env.example .env
   # Edit .env with your api key
   ```
5. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 🐳 Docker Setup
1. Define your `GEMINI_API_KEY` inside the `.env` file.
2. Build and start the container:
   ```bash
   docker-compose up --build
   ```
3. Access the app locally at: `http://localhost:8501`.

## 🧠 Architecture
1. **Ingest**: PDF parsed via PyMuPDF. Text heavily chunked using LangChain's `RecursiveCharacterTextSplitter`.
2. **Embed**: Text chunks mapped to dense vectors by Gemini's `embedding-001`.
3. **Index**: Embeddings loaded into FAISS vector store. Temporary cached copies store to `/tmp/faiss_index` for persistence over Streamlit session reruns.
4. **Query**: User queries embedded -> Similarity Search via FAISS bounds top-k contextual text blocks.
5. **Generate**: Extracted chunk bodies injected into strict LangChain prompt alongside the question.
6. **Cite**: Page-level metadata preserved at chunking is surfaced alongside the generated answers.

## ⚠️ Notes
- Does not currently support scanned (image-only) PDFs or password-protected files.
- Max PDF limit optimized natively for <20MB for fast pipeline throughput.

---
**Author**: SY BTech CS student
