from __future__ import annotations

from typing import List

try:
    from google import genai  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency guard
    genai = None  # type: ignore


class EmbeddingService:
    """Wraps the Gemini embedding model for Pinecone compatibility."""

    def __init__(self, model_name: str, api_key: str) -> None:
        if genai is None:
            raise RuntimeError("google-genai package is required for embeddings")
        if not api_key:
            raise RuntimeError("Gemini API key is required for embeddings")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def embed(self, text: str) -> List[float]:
        response = self._client.models.embed_content(model=self._model_name, contents=text)
        embeddings = getattr(response, "embeddings", None) or []
        if not embeddings:
            raise RuntimeError("Gemini returned no embeddings")
        values = getattr(embeddings[0], "values", None)
        if not values:
            raise RuntimeError("Gemini returned empty embedding values")
        return list(values)
