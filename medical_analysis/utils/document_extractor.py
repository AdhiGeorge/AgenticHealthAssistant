from pathlib import Path
from medical_analysis.extractors.pdf_extractor import extract_text_from_pdf
from medical_analysis.extractors.image_extractor import extract_text_from_image
from medical_analysis.extractors.docx_extractor import extract_text_from_docx
from medical_analysis.extractors.txt_extractor import extract_text_from_txt
from medical_analysis.extractors.csv_extractor import extract_text_from_csv
from medical_analysis.extractors.excel_extractor import extract_text_from_excel
from medical_analysis.utils.logger import get_logger

logger = get_logger(__name__)

def extract_text_from_document(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    logger.info(f"Extracting text from document: {file_path} (type: {ext})")
    try:
        if ext == '.pdf':
            return extract_text_from_pdf(file_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            return extract_text_from_image(file_path)
        elif ext == '.docx':
            return extract_text_from_docx(file_path)
        elif ext == '.txt':
            return extract_text_from_txt(file_path)
        elif ext == '.csv':
            return extract_text_from_csv(file_path)
        elif ext in ['.xlsx', '.xls']:
            return extract_text_from_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logger.error(f"Failed to extract text: {e}")
        raise 