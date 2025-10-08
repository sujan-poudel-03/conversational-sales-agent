from __future__ import annotations

import asyncio
import json
import uuid

import pytest

try:  # Ensure required runtime deps are available.
    import pydantic_settings  # type: ignore  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - env guard
    pytest.skip("pydantic-settings is required for this integration test", allow_module_level=True)  # type: ignore[arg-type]

try:
    import langgraph  # type: ignore  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - env guard
    pytest.skip("langgraph is required for this integration test", allow_module_level=True)  # type: ignore[arg-type]

from src.app.dependencies import (
    get_calendar_service,
    get_lead_service,
    get_orchestrator,
    get_rag_service,
)
from src.app.main import app
from src.orchestrator.intents import Intent
from src.orchestrator.state import ConversationState
from src.services.calendar import BookingResult, CalendarService
from src.services.lead import LeadService


class InMemoryInsertResult:
    def __init__(self, inserted_id: str) -> None:
        self.inserted_id = inserted_id


class InMemoryCollection:
    def __init__(self) -> None:
        self.inserted_documents = []

    def insert_one(self, payload):
        inserted_id = str(uuid.uuid4())
        record = {"_id": inserted_id, **payload}
        self.inserted_documents.append(record)
        return InMemoryInsertResult(inserted_id)


class StubEmailClient:
    def __init__(self) -> None:
        self.sent_messages = []

    def send(self, recipient: str, subject: str, body: str):
        message = {"recipient": recipient, "subject": subject, "body": body}
        self.sent_messages.append(message)
        return {"status": "queued", **message}


class StubCalendarService(CalendarService):
    def __init__(self) -> None:
        super().__init__(calendar_client=None)  # type: ignore[arg-type]

    def handle_booking(self, *_, **__):
        return BookingResult(appointment_id="appt_stub", message="No calendar action performed.")


class StubRagService:
    def answer_query(self, *_, **__):  # pragma: no cover - simple stub
        return "Stubbed RAG response"


class StubOrchestrator:
    def __init__(
        self,
        lead_service: LeadService,
        calendar_service: CalendarService,
        rag_service: StubRagService,
    ) -> None:
        self._lead_service = lead_service
        self._calendar_service = calendar_service
        self._rag_service = rag_service

    def run(self, state: ConversationState) -> ConversationState:
        updated = state.copy()
        updated.intent = Intent.PURCHASE_INTEREST

        rag_reply = self._rag_service.answer_query(
            context=updated.context,
            query=updated.user_query,
            history=updated.history,
        )
        updated.history.append({"role": "assistant", "content": rag_reply})

        lead_result = self._lead_service.capture_lead_step(
            context=updated.context,
            user_query=updated.user_query,
            existing_lead=updated.lead_data,
        )
        updated.lead_data.update(lead_result.updates)
        if lead_result.prompt:
            updated.history.append({"role": "assistant", "content": lead_result.prompt})

        if lead_result.completed:
            saved = self._lead_service.persist_lead(updated.context, updated.lead_data)
            confirmation = self._lead_service.build_confirmation_message(updated.lead_data)
            updated.history.append({"role": "assistant", "content": confirmation})
            booking = self._calendar_service.handle_booking(
                context=updated.context,
                user_query=updated.user_query,
                lead_data=updated.lead_data,
                appointment_id=updated.appointment_id,
                intent=updated.intent,
            )
            updated.appointment_id = booking.appointment_id
            updated.history.append({"role": "system", "content": booking.message})
            updated.history.append({"role": "system", "content": f"Lead saved: {saved['id']}"})

        return updated

    def lead_is_complete(self, lead_data):
        return self._lead_service.is_complete(lead_data)


async def _call_app(method: str, path: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("testclient", 0),
        "server": ("testserver", 80),
    }

    messages = [{"type": "http.request", "body": body, "more_body": False}]
    send_events = []

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):  # type: ignore[no-untyped-def]
        send_events.append(message)

    await app(scope, receive, send)

    status = 500
    response_body = b""
    for event in send_events:
        if event["type"] == "http.response.start":
            status = event["status"]
        elif event["type"] == "http.response.body":
            response_body += event.get("body", b"")

    data = json.loads(response_body.decode("utf-8")) if response_body else {}
    return status, data


@pytest.fixture()
def lead_dependencies_override():
    collection = InMemoryCollection()
    email_client = StubEmailClient()
    lead_service = LeadService(collection=collection, email_client=email_client)
    calendar_service = StubCalendarService()
    rag_service = StubRagService()
    orchestrator = StubOrchestrator(lead_service, calendar_service, rag_service)

    app.dependency_overrides[get_lead_service] = lambda: lead_service
    app.dependency_overrides[get_calendar_service] = lambda: calendar_service
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator

    try:
        yield collection, email_client
    finally:
        app.dependency_overrides.pop(get_lead_service, None)
        app.dependency_overrides.pop(get_calendar_service, None)
        app.dependency_overrides.pop(get_rag_service, None)
        app.dependency_overrides.pop(get_orchestrator, None)


def test_lead_capture_end_to_end(lead_dependencies_override):
    collection, email_client = lead_dependencies_override

    payload = {
        "context": {
            "org_id": "org-integration",
            "branch_id": "branch-42",
            "user_session_id": "session-lead",
        },
        "message": {
            "role": "user",
            "content": (
                "Hi there, I'm Jordan Smith. I'm interested in solar panels and battery systems "
                "because we want reliable backup power. Our budget is around $12000. "
                "You can email me at jordan.smith@example.com or call +1 555-777-1234."
            ),
        },
        "history": [],
    }

    status, data = asyncio.run(_call_app("POST", "/api/v1/chat", payload))

    assert status == 200
    assert data["intent"] == "purchase_interest"
    assert data["lead_captured"] is True
    assert "No calendar action" in data["reply"]

    # Verify lead persisted
    assert len(collection.inserted_documents) == 1
    stored = collection.inserted_documents[0]
    assert stored["org_id"] == "org-integration"
    assert stored["branch_id"] == "branch-42"
    assert stored["email"] == "jordan.smith@example.com"
    assert stored["phone"] == "+1 555-777-1234"
    assert stored["interest_reason"].lower().startswith("we want reliable backup")
    assert stored["product_interest"] == ["solar panels", "battery systems"]

    # Ensure confirmation email would be sent
    assert len(email_client.sent_messages) == 1
    sent = email_client.sent_messages[0]
    assert sent["recipient"] == "jordan.smith@example.com"
    assert "Thanks" in sent["subject"]
