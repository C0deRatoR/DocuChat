from langchain_ollama import ChatOllama
from langchain_core.documents import Document


def summarize_documents(documents: list[Document], max_chars: int = 15000) -> str:
    """
    Generates a concise summary of the uploaded document using Ollama (Llama 3.2 3B).

    Samples text evenly across chunks to stay within token limits, then asks
    the LLM for a structured markdown summary.

    Args:
        documents: List of LangChain Document chunks from the ingestion pipeline.
        max_chars:  Maximum characters of source text to feed into the prompt.

    Returns:
        A markdown-formatted summary string.
    """
    if not documents:
        return "No content available to summarize."

    # --- Sample text evenly across the document ---
    total_chunks = len(documents)
    combined_text = ""

    if total_chunks <= 10:
        # Small doc — use everything
        for doc in documents:
            combined_text += doc.page_content + "\n\n"
    else:
        # Large doc — sample ~10 evenly-spaced chunks
        step = total_chunks // 10
        for i in range(0, total_chunks, step):
            combined_text += documents[i].page_content + "\n\n"

    # Truncate to max_chars to respect token limits
    combined_text = combined_text[:max_chars]

    # --- Build and invoke the summary chain ---
    llm = ChatOllama(model="llama3.2", temperature=0)

    prompt = f"""You are a document analyst. Read the following text extracted from a PDF
and produce a concise summary in **markdown** format.

Your summary MUST include:
1. **Title / Subject** — What is this document about?
2. **Key Points** — 3-5 bullet points covering the most important information.
3. **Document Type** — (e.g., research paper, report, manual, contract, notes, etc.)

Keep the summary under 200 words. Do NOT fabricate information — only use what is provided.

---

{combined_text}

---

Summary:"""

    response = llm.invoke(prompt)
    return response.content
