from PIL import Image
import pytesseract

def extract_text_from_image(file_path: str) -> str:
    """Extract text from an image file using OCR."""
    image = Image.open(file_path)
    return pytesseract.image_to_string(image) 