import pandas as pd

def extract_text_from_excel(file_path: str) -> str:
    """Extract text from an Excel file, including all sheets."""
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    results = []
    for sheet_name in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        results.append(f"--- Sheet: {sheet_name} ---\n{df.to_string()}")
    return "\n\n".join(results) 