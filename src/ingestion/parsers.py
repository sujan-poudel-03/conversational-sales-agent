from __future__ import annotations

from typing import Iterable, List


def simple_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def parse_documents(files: Iterable[tuple[str, str]]) -> Iterable[dict]:
    for filename, content in files:
        for chunk in simple_chunk(content):
            yield {
                "text": chunk,
                "source_path": filename,
            }