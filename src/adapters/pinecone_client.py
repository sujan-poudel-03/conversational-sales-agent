from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from pinecone import Pinecone as PineconeClient  # type: ignore


@dataclass
class PineconeClientFactory:
    api_key: str
    index_name: str

    def get_index(self):  # type: ignore[override]
        if not self.api_key:
            raise RuntimeError("Pinecone API key must be configured")
        if not self.index_name:
            raise RuntimeError("Pinecone index name must be configured")
        client = PineconeClient(api_key=self.api_key)
        return client.Index(self.index_name)


class PineconeIndexProtocol:
    """Subset of pinecone.Index needed by the RAG service."""

    def query(self, *args, **kwargs) -> Dict:  # pragma: no cover - interface only
        raise NotImplementedError

    def upsert(self, vectors: List, namespace: str | None = None):  # pragma: no cover
        raise NotImplementedError
