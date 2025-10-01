from __future__ import annotations

from typing import Dict, List, Protocol

from src.adapters.pinecone_client import PineconeIndexProtocol


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> List[float]:  # pragma: no cover - interface
        ...


class RagService:
    """Handles multi-tenant retrieval over the vector store."""

    def __init__(self, pinecone_index: PineconeIndexProtocol, embedder: EmbeddingProvider) -> None:
        self._index = pinecone_index
        self._embedder = embedder

    def answer_query(self, context: Dict[str, str], query: str, history: List[Dict[str, str]]) -> str:
        filter_payload = {
            "$and": [
                {"org_id": context["org_id"]},
                {"branch_id": context["branch_id"]},
            ]
        }
        vector = self._embedder.embed(query)
        result = self._index.query(
            vector=vector,
            top_k=5,
            include_metadata=True,
            filter=filter_payload,
        )
        matches = result.get("matches", [])
        context_snippets = [match["metadata"].get("text", "") for match in matches if match.get("metadata")]
        if not context_snippets:
            return "I could not find information for that request."
        response = "\n".join(context_snippets)
        return response