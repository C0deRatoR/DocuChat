import fitz  # PyMuPDF

class PDFProcessingError(Exception):
    pass

def validate_pdf(file_bytes: bytes) -> fitz.Document:
    """
    Validates the uploaded PDF bytes and returns a PyMuPDF document if valid.
    Raises PDFProcessingError if password-protected, empty, or fails to open.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise PDFProcessingError(f"Failed to open PDF: {e}")

    if doc.is_encrypted:
        raise PDFProcessingError("PDF is password-protected. Please upload an unprotected PDF.")

    if doc.page_count == 0:
        raise PDFProcessingError("PDF is empty. Please upload a valid document.")

    return doc

def extract_pages(doc: fitz.Document) -> list[dict]:
    """
    Extracts text page by page from the document.
    Returns a list of dicts: {'page_number': int, 'text': str}
    Raises PDFProcessingError if no text can be extracted (e.g., scanned PDF).
    """
    pages = []
    total_text_length = 0
    
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()
        
        pages.append({
            "page_number": page_num + 1,  # 1-indexed
            "text": text
        })
        total_text_length += len(text)
        
    if total_text_length == 0:
        raise PDFProcessingError("No text found in the document. Image-only/scanned PDFs are not supported in this version.")
        
    return pages
