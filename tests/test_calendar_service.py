from __future__ import annotations

from datetime import datetime, timedelta

from src.adapters.calendar_client import CalendarClient
from src.adapters.email_client import EmailClient
from src.orchestrator.intents import Intent
from src.services.calendar import CalendarService


class StubCalendarClient(CalendarClient):
    def __init__(self) -> None:
        super().__init__(service_account_file="svc.json", default_timezone="UTC")
        self.created = []
        self.patched = []

    def create_event(self, calendar_id, body):
        self.created.append({"calendar_id": calendar_id, "body": body})
        return {"id": "evt-created"}

    def patch_event(self, calendar_id, event_id, body):
        self.patched.append({"calendar_id": calendar_id, "event_id": event_id, "body": body})
        return {"id": event_id}


class StubEmailClient(EmailClient):
    def __init__(self) -> None:
        super().__init__(api_key="stub", sender_domain="example.com")
        self.sent = []

    def send(self, recipient: str, subject: str, body: str):
        self.sent.append({"recipient": recipient, "subject": subject, "body": body})
        return {"status": "queued"}


def _fixed_times(service: CalendarService):
    start = datetime(2024, 1, 1, 15, 0, 0)
    end = start + timedelta(minutes=30)
    service._resolve_times = lambda *_: (start, end)  # type: ignore[method-assign]


def _context():
    return {"org_id": "org-alpha", "branch_id": "branch-east"}


def _lead():
    return {"email": "casey@example.com", "name": "Casey", "product_interest": ["solar install"]}


def test_booking_creates_event_and_sends_email():
    calendar = StubCalendarClient()
    email = StubEmailClient()
    service = CalendarService(calendar_client=calendar, email_client=email)
    _fixed_times(service)

    result = service.handle_booking(
        context=_context(),
        user_query="Book me tomorrow afternoon",
        lead_data=_lead(),
        appointment_id=None,
        intent=Intent.BOOKING,
    )

    assert result.appointment_id == "evt-created"
    assert "booked" in result.message.lower()
    assert calendar.created and calendar.created[0]["calendar_id"].startswith("org-alpha__branch-east")
    assert email.sent and email.sent[0]["subject"] == "Appointment booked"


def test_reschedule_updates_event():
    calendar = StubCalendarClient()
    email = StubEmailClient()
    service = CalendarService(calendar_client=calendar, email_client=email)
    _fixed_times(service)

    result = service.handle_booking(
        context=_context(),
        user_query="Can we reschedule next week?",
        lead_data=_lead(),
        appointment_id="evt-existing",
        intent=Intent.BOOKING,
    )

    assert result.appointment_id == "evt-existing"
    assert "rescheduled" in result.message.lower()
    assert calendar.patched and calendar.patched[0]["event_id"] == "evt-existing"
    assert email.sent and email.sent[0]["subject"] == "Appointment rescheduled"


def test_cancel_handles_missing_and_existing_event():
    calendar = StubCalendarClient()
    email = StubEmailClient()
    service = CalendarService(calendar_client=calendar, email_client=email)
    _fixed_times(service)

    missing = service.handle_booking(
        context=_context(),
        user_query="Cancel my meeting",
        lead_data=_lead(),
        appointment_id=None,
        intent=Intent.CANCEL_BOOKING,
    )
    assert missing.appointment_id is None
    assert "couldn't find" in missing.message.lower()

    confirmed = service.handle_booking(
        context=_context(),
        user_query="Cancel my meeting",
        lead_data=_lead(),
        appointment_id="evt-existing",
        intent=Intent.CANCEL_BOOKING,
    )
    assert confirmed.appointment_id == "evt-existing"
    assert "cancelled" in confirmed.message.lower()
    assert calendar.patched[-1]["event_id"] == "evt-existing"
    assert email.sent[-1]["subject"] == "Appointment cancelled"
