import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file, using OCR if direct extraction fails."""
    text_parts = []
    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if not text or len(text.strip()) < 50:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
            text_parts.append(text.strip())
    return "\n\n".join(text_parts) 