"""Utility functions for text chunking and token counting."""
from typing import List

try:
    from transformers import BertTokenizerFast
except ImportError:
    BertTokenizerFast = None

def get_default_tokenizer():
    """Return a default tokenizer (BERT). Extend as needed for other models."""
    if BertTokenizerFast is None:
        raise ImportError("transformers is not installed. Please install it to use tokenization utilities.")
    return BertTokenizerFast.from_pretrained("bert-base-uncased")

def count_tokens(text: str, tokenizer=None) -> int:
    """Count the number of tokens in the text using the provided tokenizer."""
    if tokenizer is None:
        tokenizer = get_default_tokenizer()
    return len(tokenizer.encode(text))

def chunk_text(text: str, tokenizer=None, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks based on token count.
    Args:
        text: The input text to split.
        tokenizer: The tokenizer to use (default: BERT).
        max_tokens: Maximum tokens per chunk.
        overlap: Number of tokens to overlap between chunks.
    Returns:
        List of text chunks.
    """
    if tokenizer is None:
        tokenizer = get_default_tokenizer()
    input_ids = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(input_ids):
        end = min(start + max_tokens, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(input_ids):
            break
        start += max_tokens - overlap
    return chunks 