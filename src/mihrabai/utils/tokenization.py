"""
Utilities for token counting and text manipulation
"""

from typing import Dict, List, Optional

import tiktoken


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using the specified model's tokenizer"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def truncate_text(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to fit within max_tokens"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def chunk_text(
    text: str, chunk_size: int, overlap: int = 0, model: str = "gpt-3.5-turbo"
) -> List[str]:
    """Split text into chunks of specified token size with optional overlap"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    chunks = []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(encoding.decode(chunk_tokens))

    return chunks
