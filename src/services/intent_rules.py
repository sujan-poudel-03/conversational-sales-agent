from __future__ import annotations

from src.orchestrator.intents import Intent


class RuleBasedIntentClassifier:
    """Lightweight heuristic classifier for development and testing."""

    def classify(self, state) -> Intent:
        query = state.user_query.lower()
        if any(keyword in query for keyword in ["book", "schedule", "appointment"]):
            return Intent.BOOKING
        if any(keyword in query for keyword in ["cancel", "reschedule"]):
            return Intent.CANCEL_BOOKING
        if any(keyword in query for keyword in ["interested", "buy", "price", "cost"]):
            return Intent.PURCHASE_INTEREST
        return Intent.RAG_INFO