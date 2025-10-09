from __future__ import annotations

from typing import Dict, List

from src.services.rag import RagService


class StubEmbedder:
    def __init__(self) -> None:
        self.captured: List[str] = []

    def embed(self, text: str) -> List[float]:
        self.captured.append(text)
        return [0.1, 0.2, 0.3]


class StubPineconeIndex:
    def __init__(self, documents: List[Dict]) -> None:
        self.documents = documents
        self.last_query = {}

    def query(self, *, vector, top_k, include_metadata, filter):
        self.last_query = {"vector": vector, "top_k": top_k, "filter": filter}
        matches = []
        for doc in self.documents:
            meta = doc["metadata"]
            if meta["org_id"] == filter["$and"][0]["org_id"] and meta["branch_id"] == filter["$and"][1]["branch_id"]:
                matches.append({"metadata": meta})
        return {"matches": matches[:top_k]}


def test_rag_service_filters_by_tenant():
    documents = [
        {"metadata": {"org_id": "org-a", "branch_id": "branch-1", "text": "Solar incentives overview."}},
        {"metadata": {"org_id": "org-a", "branch_id": "branch-2", "text": "Other branch data."}},
    ]
    embedder = StubEmbedder()
    index = StubPineconeIndex(documents)
    rag = RagService(pinecone_index=index, embedder=embedder)

    response = rag.answer_query(
        context={"org_id": "org-a", "branch_id": "branch-1"},
        query="Tell me about incentives",
        history=[],
    )

    assert embedder.captured == ["Tell me about incentives"]
    assert "Solar incentives overview." in response
    assert "Other branch data." not in response
