from __future__ import annotations

from typing import List

try:
    from google.generativeai import EmbeddingModel  # type: ignore
except ImportError:  # pragma: no cover
    EmbeddingModel = None


class EmbeddingService:
    """Wraps the Gemini embedding model for Pinecone compatibility."""

    def __init__(self, model_name: str, api_key: str) -> None:
        if EmbeddingModel is None:
            raise RuntimeError("google-generativeai package is required for embeddings")
        if not api_key:
            raise RuntimeError("Gemini API key is required for embeddings")
        self._model = EmbeddingModel(model_name)

    def embed(self, text: str) -> List[float]:
        response = self._model.embed_content(text)
        return response["embedding"]