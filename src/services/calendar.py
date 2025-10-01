from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

from src.adapters.calendar_client import CalendarClient
from src.orchestrator.intents import Intent


@dataclass
class BookingResult:
    appointment_id: Optional[str]
    message: str


class CalendarService:
    """Handles booking lifecycle with Google Calendar."""

    def __init__(self, calendar_client: CalendarClient) -> None:
        self._client = calendar_client

    def handle_booking(
        self,
        context: Dict[str, str],
        user_query: str,
        lead_data: Dict[str, str],
        appointment_id: Optional[str],
        intent: Intent,
    ) -> BookingResult:
        calendar_id = self._calendar_for(context)
        desired_start = datetime.utcnow() + timedelta(days=1)
        desired_end = desired_start + timedelta(minutes=30)

        event_body = {
            "summary": lead_data.get("product_interest", "Consultation"),
            "description": user_query,
            "start": {"dateTime": desired_start.isoformat(), "timeZone": self._client.default_timezone},
            "end": {"dateTime": desired_end.isoformat(), "timeZone": self._client.default_timezone},
            "attendees": self._attendees_for(lead_data),
        }

        if intent == Intent.CANCEL_BOOKING and appointment_id:
            updated = self._client.patch_event(
                calendar_id=calendar_id,
                event_id=appointment_id,
                body={"status": "cancelled"},
            )
            return BookingResult(appointment_id=updated.get("id"), message="Appointment cancelled")

        created = self._client.create_event(calendar_id=calendar_id, body=event_body)
        return BookingResult(appointment_id=created.get("id"), message="Appointment booked")

    def _calendar_for(self, context: Dict[str, str]) -> str:
        return f"{context['org_id']}__{context['branch_id']}@example.com"

    def _attendees_for(self, lead_data: Dict[str, str]):
        attendees = []
        email = lead_data.get("email")
        if email:
            attendees.append({"email": email})
        return attendees