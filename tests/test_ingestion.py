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
    vector_meta = call["vectors"][0]
    chunk_id, vector, metadata = vector_meta
    assert chunk_id, "chunk_id should be generated internally"
    assert isinstance(vector, list) and vector, "vector should be computed"
    assert metadata["org_id"] == "org_1"
    assert metadata["branch_id"] == "branch_1"
    assert metadata["source_path"].endswith("requirements.txt")
