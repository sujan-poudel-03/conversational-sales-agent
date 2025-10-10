from __future__ import annotations

from types import SimpleNamespace

from src.orchestrator.intents import Intent
from src.services.intent_rules import SemanticIntentClassifier


def _classify(text: str, classifier: SemanticIntentClassifier) -> Intent:
    state = SimpleNamespace(user_query=text)
    return classifier.classify(state)


def test_booking_queries_map_to_booking_intent():
    classifier = SemanticIntentClassifier()
    intent = _classify("Can you book an appointment for tomorrow afternoon?", classifier)
    assert intent is Intent.BOOKING


def test_reschedule_queries_map_to_cancel_intent():
    classifier = SemanticIntentClassifier()
    intent = _classify("We need to reschedule our meeting to next week.", classifier)
    assert intent is Intent.CANCEL_BOOKING


def test_pricing_questions_map_to_purchase_interest():
    classifier = SemanticIntentClassifier()
    intent = _classify("I'm interested in your plans—how much do they cost?", classifier)
    assert intent is Intent.PURCHASE_INTEREST


def test_general_questions_default_to_rag():
    classifier = SemanticIntentClassifier()
    intent = _classify("What kinds of services do you offer?", classifier)
    assert intent is Intent.RAG_INFO
