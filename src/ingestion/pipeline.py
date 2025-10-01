from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable, List

from src.adapters.pinecone_client import PineconeIndexProtocol


@dataclass
class IngestionChunk:
    id: str
    text: str
    metadata: dict


class IngestionPipeline:
    """Processes documents into Pinecone with tenant metadata."""

    def __init__(self, pinecone_index: PineconeIndexProtocol) -> None:
        self._index = pinecone_index

    def run(self, *, context: dict, documents: Iterable[dict]) -> dict:
        vectors: List = []
        for doc in documents:
            chunk_id = doc.get("chunk_id") or str(uuid.uuid4())
            metadata = {
                "org_id": context["org_id"],
                "branch_id": context["branch_id"],
                "source_file": doc.get("source_file"),
                "text": doc.get("text"),
            }
            vectors.append((chunk_id, doc["vector"], metadata))

        if not vectors:
            return {"processed": 0, "failed": 0}

        self._index.upsert(vectors=vectors)
        return {"processed": len(vectors), "failed": 0}