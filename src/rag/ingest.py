from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.pdf_utils import validate_pdf, extract_pages, PDFProcessingError

def process_pdf(file_bytes: bytes, chunk_size: int = 2000, chunk_overlap: int = 200) -> list[Document]:
    """
    End-to-end processing of a PDF file upload:
    1. Validate PDF
    2. Extract text with page numbers
    3. Split into manageable chunks
    
    Returns a list of LangChain Document objects containing the chunk text and page metadata.
    """
    doc = validate_pdf(file_bytes)
    
    try:
        pages = extract_pages(doc)
    finally:
        doc.close()
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents = []
    for page in pages:
        # Ignore pages that are effectively empty even if no overall error was thrown
        if not page["text"]:
            continue
            
        chunks = text_splitter.split_text(page["text"])
        
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"page_number": page["page_number"]}
                )
            )
            
    return documents
