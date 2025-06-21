from docx import Document

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs]) 