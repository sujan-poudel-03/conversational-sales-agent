from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:  # Newer Pinecone SDK (>=3)
    from pinecone import Pinecone as PineconeClient  # type: ignore
except ImportError:  # pragma: no cover - fallback to legacy SDK
    PineconeClient = None  # type: ignore

try:
    import pinecone  # type: ignore  # legacy module entry point
except ImportError:  # pragma: no cover - pinecone optional during dev
    pinecone = None


@dataclass
class PineconeClientFactory:
    api_key: str
    environment: str
    index_name: str

    def get_index(self):  # type: ignore[override]
        """Return a Pinecone index instance compatible with the active SDK."""

        if PineconeClient is not None:
            if not self.api_key:
                raise RuntimeError("Pinecone API key must be configured")
            client = PineconeClient(api_key=self.api_key)
            return client.Index(self.index_name)

        if not pinecone:
            raise RuntimeError("pinecone package is required to use RagService")
        if not self.environment:
            raise RuntimeError("Pinecone environment must be configured for legacy SDK usage")
        pinecone.init(api_key=self.api_key, environment=self.environment)
        return pinecone.Index(self.index_name)


class PineconeIndexProtocol:
    """Subset of pinecone.Index needed by the RAG service."""

    def query(self, *args, **kwargs) -> Dict:  # pragma: no cover - interface only
        raise NotImplementedError

    def upsert(self, vectors: List, namespace: str | None = None):  # pragma: no cover
        raise NotImplementedError
