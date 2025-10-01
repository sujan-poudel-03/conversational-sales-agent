from __future__ import annotations

from typing import Dict

from google.generativeai import GenerativeModel  # type: ignore

from src.orchestrator.intents import Intent


class IntentClassifier:
    """LLM-backed intent classifier using Gemini."""

    def __init__(self, model_name: str, api_key: str) -> None:
        if not api_key:
            raise RuntimeError("Gemini API key is required for intent classification")
        self._model = GenerativeModel(model_name=model_name)

    def classify(self, state) -> Intent:
        prompt = (
            "Classify the user intent into one of RAG_INFO, PURCHASE_INTEREST, BOOKING,"
            " or CANCEL_BOOKING. Only return the label. Query: "
            f"{state.user_query}"
        )
        response = self._model.generate_content(prompt)
        label = response.text.strip().upper()
        return Intent.from_label(label)