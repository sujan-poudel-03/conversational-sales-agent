from __future__ import annotations

try:
    from google import genai  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency guard
    genai = None  # type: ignore

from src.orchestrator.intents import Intent


class IntentClassifier:
    """LLM-backed intent classifier using Gemini."""

    def __init__(self, model_name: str, api_key: str) -> None:
        if genai is None:
            raise RuntimeError("google-genai package is required for intent classification")
        if not api_key:
            raise RuntimeError("Gemini API key is required for intent classification")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def classify(self, state) -> Intent:
        prompt = (
            "Classify the user intent into one of RAG_INFO, PURCHASE_INTEREST, BOOKING,"
            " or CANCEL_BOOKING. Only return the label. Query: "
            f"{state.user_query}"
        )
        response = self._client.models.generate_content(model=self._model_name, contents=prompt)
        text = self._extract_text(response)
        label = text.strip().upper()
        return Intent.from_label(label)

    def _extract_text(self, response) -> str:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or [] if content else []
            for part in parts:
                value = getattr(part, "text", None)
                if value:
                    return value
        raise RuntimeError("Gemini did not return text for intent classification")
