from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Iterable, List, Protocol, Sequence, Tuple

from src.adapters.pinecone_client import PineconeIndexProtocol
from src.ingestion.parsers import simple_chunk

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> List[float]:
        ...


VectorDict = dict
VectorLegacy = Tuple[str, List[float], dict]


class IngestionPipeline:
    """Processes documents into Pinecone with tenant metadata."""

    def __init__(
        self,
        pinecone_index: PineconeIndexProtocol,
        embedder: EmbeddingProvider,
        base_path: Path | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self._index = pinecone_index
        self._embedder = embedder
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def run(self, *, context: dict, documents: Iterable[dict]) -> dict:
        processed = 0
        failed = 0
        vectors_modern: List[VectorDict] = []
        vectors_legacy: List[VectorLegacy] = []

        for document in documents:
            try:
                resolved_chunks = list(self._prepare_chunks(document))
                if not resolved_chunks:
                    logger.warning("No content extracted from document", extra={"document": document})
                    continue

                for chunk in resolved_chunks:
                    text = chunk["text"]
                    metadata = {
                        "org_id": context.get("org_id"),
                        "branch_id": context.get("branch_id"),
                        "session_id": context.get("user_session_id"),
                        "source_path": chunk.get("source_path"),
                        "text": text,
                    }
                    values = self._embedder.embed(text)
                    vectors_modern.append(
                        {
                            "id": chunk["chunk_id"],
                            "values": values,
                            "metadata": metadata,
                        }
                    )
                    vectors_legacy.append((chunk["chunk_id"], values, metadata))
                    processed += 1
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to ingest document", extra={"document": document})
                failed += 1

        if not vectors_modern:
            return {"processed": processed, "failed": failed}

        namespace = self._build_namespace(context)
        baseline_count = self._namespace_vector_count(namespace)
        self._upsert(namespace, vectors_modern, vectors_legacy)
        self._await_vector_count(namespace, baseline_count + len(vectors_modern))
        return {"processed": processed, "failed": failed}

    def _prepare_chunks(self, document: dict) -> Iterable[dict]:
        text = document.get("text")
        source_path = document.get("source_path") or document.get("source_file")

        if text:
            yield from self._chunk_text(text=text, source_path=source_path)
            return

        if not source_path:
            raise ValueError("Document must provide either 'text' or 'source_path'.")

        file_content = self._load_file(source_path)
        yield from self._chunk_text(text=file_content, source_path=source_path)

    def _chunk_text(self, *, text: str, source_path: str | None) -> Iterable[dict]:
        for chunk in simple_chunk(text, chunk_size=self._chunk_size, overlap=self._chunk_overlap):
            yield {
                "chunk_id": str(uuid.uuid4()),
                "text": chunk,
                "source_path": source_path,
            }

    def _load_file(self, source_path: str) -> str:
        candidate = Path(source_path)
        if not candidate.is_absolute():
            candidate = (self._base_path / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        return candidate.read_text(encoding="utf-8")

    def _upsert(
        self,
        namespace: str,
        modern_vectors: Sequence[VectorDict],
        legacy_vectors: Sequence[VectorLegacy],
    ) -> None:
        try:
            self._index.upsert(vectors=list(modern_vectors), namespace=namespace)
            return
        except TypeError:
            pass
        except Exception as exc:
            if not self._should_retry_with_legacy_format(exc):
                raise
        self._index.upsert(vectors=list(legacy_vectors), namespace=namespace)

    def _await_vector_count(self, namespace: str, target_count: int) -> None:
        describe = getattr(self._index, "describe_index_stats", None)
        if not callable(describe):
            return

        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                stats = describe()
            except Exception:
                time.sleep(0.5)
                continue
            namespaces = stats.get("namespaces", {}) if isinstance(stats, dict) else {}
            current = int(namespaces.get(namespace, {}).get("vector_count", 0))
            if current >= target_count:
                return
            time.sleep(0.5)

    def _namespace_vector_count(self, namespace: str) -> int:
        describe = getattr(self._index, "describe_index_stats", None)
        if not callable(describe):
            return 0
        try:
            stats = describe()
        except Exception:
            return 0
        namespaces = stats.get("namespaces", {}) if isinstance(stats, dict) else {}
        return int(namespaces.get(namespace, {}).get("vector_count", 0))

    @staticmethod
    def _should_retry_with_legacy_format(exc: Exception) -> bool:
        message = str(exc)
        fallback_markers = (
            "Vectors item must be a dict",
            "Expected List[Tuple",
            "Vectors should contain",
        )
        return any(marker in message for marker in fallback_markers)

    @staticmethod
    def _build_namespace(context: dict) -> str:
        org_id = context.get("org_id", "default_org")
        branch_id = context.get("branch_id", "default_branch")
        return f"{org_id}::{branch_id}"
