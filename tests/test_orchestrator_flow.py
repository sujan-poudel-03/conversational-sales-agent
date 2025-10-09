from __future__ import annotations

from dataclasses import dataclass

from src.orchestrator.graph import AgentOrchestrator
from src.orchestrator.intents import Intent
from src.orchestrator.state import ConversationState
from src.services.calendar import BookingResult, CalendarService
from src.services.lead import LeadCaptureResult, LeadService
from src.services.rag import RagService


class MemoryCollection:
    def __init__(self) -> None:
        self.inserted = []

    def insert_one(self, payload):
        self.inserted.append(payload)
        return type("Result", (), {"inserted_id": "lead-123"})


class MemoryEmailClient:
    def __init__(self) -> None:
        self.sent = []

    def send(self, recipient: str, subject: str, body: str):
        self.sent.append({"recipient": recipient, "subject": subject, "body": body})
        return {"status": "queued"}


class StubRagService(RagService):
    def __init__(self) -> None:
        self.answered = []

    def answer_query(self, context, query, history):
        self.answered.append({"context": context, "query": query, "history": history})
        return "Here is the knowledge base answer."


@dataclass
class StubLeadCaptureResult:
    updates: dict
    prompt: str | None = None
    completed: bool = False


class StubLeadService(LeadService):
    def __init__(self) -> None:
        super().__init__(collection=MemoryCollection(), email_client=MemoryEmailClient())
        self.capture_calls = 0

    def capture_lead_step(self, context, user_query, existing_lead):
        self.capture_calls += 1
        return LeadCaptureResult(updates={"user_query": user_query}, prompt="Captured.", completed=False)

    def is_complete(self, lead_data):
        required = {"name", "email", "product_interest", "interest_reason"}
        return required.issubset({k for k, v in lead_data.items() if v})


class StubCalendarService(CalendarService):
    def __init__(self):
        self.operations = []

    def handle_booking(self, context, user_query, lead_data, appointment_id, intent):
        self.operations.append(
            {
                "context": context,
                "user_query": user_query,
                "lead_data": lead_data,
                "appointment_id": appointment_id,
                "intent": intent,
            }
        )
        return BookingResult(appointment_id="appt-001", message="Booking confirmed.", audit_note="calendar_event_created:appt-001")


def _build_orchestrator(intent_picker):
    rag = StubRagService()
    lead = LeadService(collection=MemoryCollection(), email_client=MemoryEmailClient())
    calendar = StubCalendarService()
    orchestrator = AgentOrchestrator(
        rag_service=rag,
        lead_service=lead,
        calendar_service=calendar,
        intent_classifier=intent_picker,
    )
    return orchestrator, rag, lead, calendar


def test_rag_intent_routes_to_rag_node():
    orchestrator, rag, _, _ = _build_orchestrator(lambda state: Intent.RAG_INFO)
    state = ConversationState(
        intent=Intent.RAG_INFO,
        user_query="Tell me about financing options.",
        context={"org_id": "org", "branch_id": "branch"},
    )

    final_state = orchestrator.run(state)

    assert final_state.intent is Intent.RAG_INFO
    assert final_state.history[-1]["content"] == "Here is the knowledge base answer."
    assert rag.answered


def test_purchase_intent_prioritises_lead_capture():
    orchestrator, rag, lead, _ = _build_orchestrator(lambda state: Intent.PURCHASE_INTEREST)
    state = ConversationState(
        intent=Intent.RAG_INFO,
        user_query="I'm interested in solar panels and batteries.",
        context={"org_id": "org", "branch_id": "branch"},
    )

    final_state = orchestrator.run(state)

    assert final_state.intent is Intent.PURCHASE_INTEREST
    assert rag.answered == []
    assert lead.is_complete(final_state.lead_data) is False
    assert final_state.history[-1]["role"] == "assistant"


def test_booking_without_complete_lead_waits_for_details():
    orchestrator, rag, lead, calendar = _build_orchestrator(lambda state: Intent.BOOKING)
    state = ConversationState(
        intent=Intent.RAG_INFO,
        user_query="Can we book next week?",
        context={"org_id": "org", "branch_id": "branch"},
        lead_data={},
    )

    final_state = orchestrator.run(state)

    assert final_state.intent is Intent.BOOKING
    assert final_state.appointment_id is None
    assert calendar.operations == []
    assert rag.answered == []
    assert final_state.history[-1]["role"] == "assistant"


def test_booking_with_complete_lead_triggers_calendar():
    orchestrator, rag, lead, calendar = _build_orchestrator(lambda state: Intent.BOOKING)
    complete_lead = {
        "name": "Skyler",
        "email": "skyler@example.com",
        "product_interest": ["solar"],
        "interest_reason": "To cut bills",
        "phone": "+1 555 222 9999",
    }
    state = ConversationState(
        intent=Intent.RAG_INFO,
        user_query="Book me tomorrow afternoon.",
        context={"org_id": "org", "branch_id": "branch"},
        lead_data=complete_lead,
    )

    final_state = orchestrator.run(state)

    assert final_state.intent is Intent.BOOKING
    assert final_state.appointment_id == "appt-001"
    assert calendar.operations and calendar.operations[0]["lead_data"]["email"] == "skyler@example.com"
    assert final_state.history[-1]["role"] == "system"
    assert "calendar_event_created" in final_state.history[-1]["content"]
