from __future__ import annotations

from types import SimpleNamespace

from src.orchestrator.graph import AgentOrchestrator
from src.orchestrator.intents import Intent
from src.orchestrator.state import ConversationState
from src.services.calendar import BookingResult
from src.services.intent_rules import RuleBasedIntentClassifier
from src.services.lead import LeadService


class FakeCollection:
    def insert_one(self, payload):
        return SimpleNamespace(inserted_id="fake-id")


class FakeEmailClient:
    def send(self, recipient: str, subject: str, body: str) -> None:  # pragma: no cover - no-op
        return None


class RecordingRagService:
    def __init__(self) -> None:
        self.was_called = False

    def answer_query(self, *args, **kwargs):  # pragma: no cover - defensive branch
        self.was_called = True
        return "stubbed response"


class StubCalendarService:
    def handle_booking(self, *_, **__):
        return BookingResult(appointment_id=None, message="No calendar action performed.")


def test_purchase_intent_takes_lead_path_before_rag():
    lead_service = LeadService(collection=FakeCollection(), email_client=FakeEmailClient())
    rag_service = RecordingRagService()
    calendar_service = StubCalendarService()
    classifier = RuleBasedIntentClassifier()

    orchestrator = AgentOrchestrator(
        rag_service=rag_service,
        lead_service=lead_service,
        calendar_service=calendar_service,
        intent_classifier=classifier.classify,
    )

    state = ConversationState(
        intent=Intent.RAG_INFO,
        user_query="I'm interested in solar panels for my home.",
        context={"org_id": "org-test", "branch_id": "branch-test"},
    )

    final_state = orchestrator.run(state)

    assert final_state.intent is Intent.PURCHASE_INTEREST
    assert rag_service.was_called is False
    assert final_state.history, "lead path should produce assistant guidance"
    assert final_state.history[0]["role"] == "assistant"
    assert "good fit for you right now" in final_state.history[0]["content"]
