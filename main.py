"""Main FastAPI application for the Medical Analysis API."""

import os
from pathlib import Path
from medical_analysis.agents.orchestrator import OrchestratorAgent
from medical_analysis.utils.db import init_db, get_analysis, save_analysis
from medical_analysis.utils.logger import get_logger
from medical_analysis.utils.config import get_config
from medical_analysis.utils.text_utils import chunk_text, get_default_tokenizer
from medical_analysis.extractors.pdf_extractor import extract_text_from_pdf
from medical_analysis.extractors.image_extractor import extract_text_from_image
from medical_analysis.extractors.docx_extractor import extract_text_from_docx
from medical_analysis.extractors.txt_extractor import extract_text_from_txt
from medical_analysis.extractors.csv_extractor import extract_text_from_csv
from medical_analysis.extractors.excel_extractor import extract_text_from_excel
import uuid

class ReviewAgent:
    """Reflection agent to compare analysis to original data and score relevancy."""
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.logger = get_logger("ReviewAgent")

    def score_relevancy(self, original, analysis):
        # Simple scoring: ratio of overlapping words (can be replaced with embedding similarity)
        original_words = set(original.lower().split())
        analysis_words = set(analysis.lower().split())
        overlap = len(original_words & analysis_words)
        score = overlap / max(len(original_words), 1)
        self.logger.info(f"Relevancy score: {score:.2f}")
        return score

    def review(self, original, analysis):
        score = self.score_relevancy(original, analysis)
        status = "approved" if score >= self.threshold else "retry"
        return status, score

def extract_text_from_document(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
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

def main():
    logger = get_logger("main")
    config = get_config()
    init_db()
    logger.info("Medical Diagnostics CLI started.")
    print("1. TEXT\n2. DOCUMENT")
    choice = input("Choose an option (1 for TEXT, 2 for DOCUMENT): ").strip()
    if choice == '1':
        print("Enter your text (press Enter on an empty line to finish):")
        lines = []
        while True:
            line = input()
            if line == '':
                break
            lines.append(line)
        text = '\n'.join(lines)
        file_path = None
    elif choice == '2':
        file_path = input("Enter the path to the document: ").strip()
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            logger.error(f"File not found: {file_path}")
            return
        try:
            text = extract_text_from_document(file_path)
        except Exception as e:
            print(f"Error extracting text: {e}")
            logger.error(f"Error extracting text: {e}")
            return
    else:
        print("Invalid choice.")
        logger.warning("Invalid choice entered.")
        return

    # Chunk and summarize if needed (before orchestrator)
    tokenizer = get_default_tokenizer()
    max_tokens = config['models']['max_tokens']
    overlap = config['models']['chunk_overlap']
    chunks = chunk_text(text, tokenizer=tokenizer, max_tokens=max_tokens, overlap=overlap)
    logger.info(f"Text split into {len(chunks)} chunk(s) for analysis.")
    aggregated_text = '\n\n'.join(chunks)

    orchestrator = OrchestratorAgent()
    session_id = str(uuid.uuid4())
    review_agent = ReviewAgent(threshold=config.get('review_threshold', 0.7))
    attempt = 0
    while True:
        attempt += 1
        logger.info(f"Analysis attempt {attempt}")
        logger.info("Processing input as document.")
        report = orchestrator.orchestrate(aggregated_text)
        # Review the analysis
        status, score = review_agent.review(text, report)
        save_analysis(session_id, text, report, status, score)
        logger.info(f"Review status: {status}, score: {score:.2f}")
        if status == "approved":
            break
        logger.warning("Analysis did not meet relevancy threshold. Retrying...")
    print("\n==== FINAL ANALYSIS REPORT ====")
    print(report)
    logger.info("Report presented to user.")

if __name__ == "__main__":
    main()
