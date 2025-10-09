from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

from src.adapters.calendar_client import CalendarClient
from src.adapters.email_client import EmailClient
from src.orchestrator.intents import Intent


@dataclass
class BookingResult:
    appointment_id: Optional[str]
    message: str
    audit_note: Optional[str] = None


class CalendarService:
    """Handles booking lifecycle with Google Calendar."""

    def __init__(self, calendar_client: CalendarClient, email_client: EmailClient) -> None:
        self._client = calendar_client
        self._email = email_client

    def handle_booking(
        self,
        context: Dict[str, str],
        user_query: str,
        lead_data: Dict[str, str],
        appointment_id: Optional[str],
        intent: Intent,
    ) -> BookingResult:
        calendar_id = self._calendar_for(context)
        desired_start, desired_end = self._resolve_times(user_query)
        event_body = {
            "summary": self._summary_for(lead_data),
            "description": user_query,
            "start": {"dateTime": desired_start.isoformat(), "timeZone": self._client.default_timezone},
            "end": {"dateTime": desired_end.isoformat(), "timeZone": self._client.default_timezone},
            "attendees": self._attendees_for(lead_data),
        }

        if intent == Intent.CANCEL_BOOKING:
            if not appointment_id:
                return BookingResult(appointment_id=None, message="I couldn't find an appointment to cancel.")
            updated = self._client.patch_event(
                calendar_id=calendar_id,
                event_id=appointment_id,
                body={"status": "cancelled"},
            )
            self._send_confirmation(lead_data, "Appointment cancelled", event_body)
            return BookingResult(
                appointment_id=updated.get("id"),
                message="Your appointment has been cancelled. Check your email for confirmation.",
                audit_note=f"calendar_event_cancelled:{updated.get('id')}",
            )

        if appointment_id:
            updated = self._client.patch_event(
                calendar_id=calendar_id,
                event_id=appointment_id,
                body={**event_body, "status": "confirmed"},
            )
            self._send_confirmation(lead_data, "Appointment rescheduled", event_body)
            return BookingResult(
                appointment_id=updated.get("id"),
                message="All set—your appointment has been rescheduled. Check your email for the details.",
                audit_note=f"calendar_event_rescheduled:{updated.get('id')}",
            )

        created = self._client.create_event(calendar_id=calendar_id, body=event_body)
        self._send_confirmation(lead_data, "Appointment booked", event_body)
        return BookingResult(
            appointment_id=created.get("id"),
            message="Your consultation is booked! I sent a confirmation email with the calendar invite.",
            audit_note=f"calendar_event_created:{created.get('id')}",
        )

    def _calendar_for(self, context: Dict[str, str]) -> str:
        org_id = context.get("org_id", "default_org")
        branch_id = context.get("branch_id", "default_branch")
        return f"{org_id}__{branch_id}@example.com"

    def _attendees_for(self, lead_data: Dict[str, str]):
        attendees = []
        email = lead_data.get("email")
        if email:
            attendees.append({"email": email})
        return attendees

    def _summary_for(self, lead_data: Dict[str, str]) -> str:
        product_interest = lead_data.get("product_interest")
        if isinstance(product_interest, list):
            items = ", ".join(product_interest)
        elif product_interest:
            items = str(product_interest)
        else:
            items = "Consultation"
        name = lead_data.get("name")
        if name:
            return f"{items} with {name}"
        return items

    def _resolve_times(self, user_query: str) -> tuple[datetime, datetime]:
        base = datetime.utcnow()
        lower = user_query.lower()
        if "tomorrow" in lower:
            start = base + timedelta(days=1)
        elif "next week" in lower:
            start = base + timedelta(days=7)
        else:
            start = base + timedelta(days=1)
        start = start.replace(hour=15, minute=0, second=0, microsecond=0)
        end = start + timedelta(minutes=30)
        return start, end

    def _send_confirmation(self, lead_data: Dict[str, str], subject: str, event_body: Dict[str, Dict[str, str]]) -> None:
        email = lead_data.get("email")
        if not email:
            return
        start = event_body["start"]["dateTime"]
        timezone = event_body["start"]["timeZone"]
        summary = event_body["summary"]
        body = (
            f"Hi {lead_data.get('name', 'there')},\n\n"
            f"{subject} for {summary} on {start} ({timezone}).\n"
            "Reply to this email if you need any changes.\n"
        )
        self._email.send(recipient=email, subject=subject, body=body)
