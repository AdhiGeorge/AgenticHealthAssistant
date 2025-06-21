import pandas as pd

def extract_text_from_csv(file_path: str) -> str:
    """Extract text from a CSV file as a string."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, engine='python')
            return df.to_string()
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(file_path, encoding='utf-8', errors='replace', engine='python')
    return df.to_string() 