from __future__ import annotations

import pytest

from src.orchestrator.intents import Intent
from src.orchestrator.state import ConversationState
from src.services.intent_rules import RuleBasedIntentClassifier


def test_classifier_uses_llm_when_available() -> None:
    calls = []

    def llm_classifier(state) -> Intent:
        calls.append(state)
        return Intent.BOOKING

    classifier = RuleBasedIntentClassifier(llm_classifier=llm_classifier)
    state = ConversationState(user_query="Anything here")

    result = classifier.classify(state)
    print(f"LLM classifier result: {result}")

    assert result == Intent.BOOKING
    assert calls, "LLM classifier should have been invoked"


def test_classifier_falls_back_to_rules_when_llm_fails() -> None:
    def failing_llm(state) -> Intent:
        raise RuntimeError("LLM failure")

    classifier = RuleBasedIntentClassifier(llm_classifier=failing_llm)
    state = ConversationState(user_query="I'm interested in the premium package")

    result = classifier.classify(state)

    assert result == Intent.PURCHASE_INTEREST


@pytest.mark.parametrize(
    "query, expected_intent",
    [
        ("Can you book me an appointment for tomorrow?", Intent.BOOKING),
        ("I need to cancel my appointment next week", Intent.CANCEL_BOOKING),
        ("I'm interested in buying your premium plan", Intent.PURCHASE_INTEREST),
        ("Tell me about your service offerings", Intent.RAG_INFO),
        ("Do you offer pricing for enterprise teams?", Intent.PURCHASE_INTEREST),
        ("Can I reschedule the meeting to Friday?", Intent.CANCEL_BOOKING),
    ],
)
def test_rule_based_intent_classifier(query: str, expected_intent: Intent) -> None:
    classifier = RuleBasedIntentClassifier()
    state = ConversationState(user_query=query)

    result = classifier.classify(state)

    assert result == expected_intent
