from __future__ import annotations

from typing import Callable, Optional

from src.orchestrator.intents import Intent


class RuleBasedIntentClassifier:
    """Lightweight heuristic classifier for development and testing."""

    def __init__(self, llm_classifier: Optional[Callable] = None) -> None:
        self._llm_classifier = llm_classifier

    def classify(self, state) -> Intent:
        if self._llm_classifier:
            try:
                return self._llm_classifier(state)
            except Exception:
                # Fall back to heuristics when the LLM path fails.
                pass

        query = state.user_query.lower()
        if self._matches_any(query, ["cancel", "reschedule", "move my appointment", "change my booking"]):
            return Intent.CANCEL_BOOKING
        if self._matches_any(query, ["book", "schedule", "appointment", "reserve", "set up a meeting"]):
            return Intent.BOOKING
        if self._matches_any(
            query,
            [
                "interested",
                "buy",
                "purchase",
                "pricing",
                "price",
                "cost",
                "quote",
                "plan",
                "package",
            ],
        ):
            return Intent.PURCHASE_INTEREST
        return Intent.RAG_INFO

    def _matches_any(self, query: str, keywords: list[str]) -> bool:
        return any(keyword in query for keyword in keywords)
