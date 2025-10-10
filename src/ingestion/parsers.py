from __future__ import annotations

from typing import Iterable, List


def simple_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    step = chunk_size - overlap
    if step <= 0:
        # Fallback to non-overlapping chunks to avoid infinite loops
        step = chunk_size

    start = 0
    total_words = len(words)
    while start < total_words:
        end = min(total_words, start + chunk_size)
        print(f"Chunk will include words {start} to {end}")
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if end >= total_words:
            break
        start += step
    return chunks


def parse_documents(files: Iterable[tuple[str, str]]) -> Iterable[dict]:
    for filename, content in files:
        for chunk in simple_chunk(content):
            yield {
                "text": chunk,
                "source_path": filename,
            }
