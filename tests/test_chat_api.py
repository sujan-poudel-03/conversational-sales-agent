from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.app.dependencies import get_orchestrator
from src.app.main import app
from src.orchestrator.graph import AgentOrchestrator
from src.orchestrator.intents import Intent
from src.services.lead import LeadService
from src.services.rag import RagService


class MemoryCollection:
    def __init__(self) -> None:
        self.inserted = []

    def insert_one(self, payload):
        self.inserted.append(payload)
        return type("InsertResult", (), {"inserted_id": "lead-789"})


class MemoryEmailClient:
    def __init__(self) -> None:
        self.sent = []

    def send(self, recipient: str, subject: str, body: str):
        self.sent.append({"recipient": recipient, "subject": subject, "body": body})
        return {"status": "queued"}


class InlineRagService(RagService):
    def answer_query(self, context, query, history):
        return "Stub knowledge base answer."


def _build_orchestrator():
    lead_service = LeadService(collection=MemoryCollection(), email_client=MemoryEmailClient())

    class DummyCalendar(CalendarService):
        def __init__(self):
            self.calls = []

        def handle_booking(self, context, user_query, lead_data, appointment_id, intent):
            self.calls.append(
                (context, user_query, lead_data, appointment_id, intent),
            )
            return type(
                "Result",
                (),
                {
                    "appointment_id": "appt-test",
                    "message": "Appointment confirmed for your request.",
                    "audit_note": "calendar_event_created:appt-test",
                },
            )()

    calendar_service = DummyCalendar()

    class StubRag(RagService):
        def answer_query(self, context, query, history):
            return "Stub knowledge base answer."

    orchestrator = AgentOrchestrator(
        rag_service=StubRag(),
        lead_service=lead_service,
        calendar_service=calendar_service,
        intent_classifier=lambda state: _classify(state.user_query),
    )
    return orchestrator, lead_service, calendar_service


def _classify(query: str) -> Intent:
    lowered = query.lower()
    if any(word in lowered for word in ["book", "schedule", "appointment"]):
        return Intent.BOOKING
    if "cancel" in lowered:
        return Intent.CANCEL_BOOKING
    if any(word in lowered for word in ["interested", "buy", "price", "quote"]):
        return Intent.PURCHASE_INTEREST
    return Intent.RAG_INFO


@pytest.fixture()
def client():
    orchestrator, lead_service, calendar_service = _build_orchestrator()
    overrides = {
        get_orchestrator: lambda: orchestrator,
    }
    app.dependency_overrides.update(overrides)
    try:
        yield TestClient(app)
    finally:
        for key in overrides:
            app.dependency_overrides.pop(key, None)


def _base_payload(content: str):
    return {
        "context": {
            "org_id": "org-int",
            "branch_id": "branch-1",
            "user_session_id": "session-1",
        },
        "message": {"role": "user", "content": content},
        "history": [],
    }


def test_chat_rag_info_flow(client: TestClient):
    payload = _base_payload("What financing options do you support?")
    response = client.post("/api/v1/chat", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["intent"] == "rag_info"
    assert "knowledge base answer" in data["reply"]


def test_chat_purchase_lead_flow(client: TestClient):
    payload = _base_payload(
        "I'm interested in solar panels for my home. Email me at jordan@example.com."
    )
    response = client.post("/api/v1/chat", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["intent"] == "purchase_interest"
    assert data["lead_captured"] is False
    assert "share more" in data["reply"].lower() or "could you" in data["reply"].lower()


def test_chat_booking_flow(client: TestClient):
    payload = _base_payload(
        "Please book an appointment for tomorrow. My name is Alex Rivera, "
        "email alex@example.com, phone +1 555 333 4444, interested in solar panels because of high bills."
    )
    response = client.post("/api/v1/chat", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert data["intent"] == "booking"
    assert data["lead_captured"] is True
    assert data["appointment_id"] == "appt-test"
    assert "appointment confirmed" in data["reply"].lower()
