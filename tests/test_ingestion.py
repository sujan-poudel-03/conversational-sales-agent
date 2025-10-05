from pathlib import Path

from src.ingestion.pipeline import IngestionPipeline
from src.services.embeddings_fallback import DeterministicEmbedding


class FakePineconeIndex:
    def __init__(self) -> None:
        self.calls = []

    def upsert(self, *, vectors, namespace=None):
        self.calls.append({"vectors": vectors, "namespace": namespace})


def test_pipeline_reads_source_file_and_upserts_vectors():
    index = FakePineconeIndex()
    embedder = DeterministicEmbedding()
    pipeline = IngestionPipeline(pinecone_index=index, embedder=embedder, base_path=Path.cwd(), chunk_size=50)

    context = {"org_id": "org_1", "branch_id": "branch_1", "user_session_id": "session_1"}
    payload = [{"source_path": "requirements.txt"}]

    result = pipeline.run(context=context, documents=payload)

    assert result["processed"] > 0
    assert result["failed"] == 0
    assert index.calls, "expected vectors to be upserted"
    call = index.calls[0]
    assert call["namespace"] == "org_1::branch_1"
    vector_item = call["vectors"][0]
    assert isinstance(vector_item, dict)
    assert vector_item["metadata"]["org_id"] == "org_1"
    assert vector_item["metadata"]["branch_id"] == "branch_1"
    assert vector_item["metadata"]["source_path"].endswith("requirements.txt")
    assert vector_item["values"], "vector should contain embedding values"
