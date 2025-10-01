from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:
    import pinecone  # type: ignore
except ImportError:  # pragma: no cover - pinecone optional during dev
    pinecone = None


@dataclass
class PineconeClientFactory:
    api_key: str
    environment: str
    index_name: str

    def get_index(self):  # type: ignore[override]
        if not pinecone:
            raise RuntimeError("pinecone package is required to use RagService")
        pinecone.init(api_key=self.api_key, environment=self.environment)
        return pinecone.Index(self.index_name)


class PineconeIndexProtocol:
    """Subset of pinecone.Index needed by the RAG service."""

    def query(self, *args, **kwargs) -> Dict:  # pragma: no cover - interface only
        raise NotImplementedError

    def upsert(self, vectors: List, namespace: str | None = None):  # pragma: no cover
        raise NotImplementedError