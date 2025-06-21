def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a TXT file, trying multiple encodings."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(file_path, 'rb') as f:
        return f.read().decode('utf-8', errors='replace') 