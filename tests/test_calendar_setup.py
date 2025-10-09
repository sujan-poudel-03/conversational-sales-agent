from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.adapters.calendar_client import CalendarClient
from src.app.config import get_settings

def _debug_hint(message: str) -> None:
    print(f"[calendar-test] {message}")


try:
    from google.oauth2 import service_account  # type: ignore  # noqa: F401
    from googleapiclient.discovery import build  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - env guard
    _debug_hint("Missing dependency: install google-api-python-client and google-auth libraries.")
    pytest.skip(
        "google-api-python-client is required for Google Calendar integration tests",
        allow_module_level=True,
    )  # type: ignore[arg-type]


settings = get_settings()
service_account_file = settings.google_service_account_file
if not service_account_file:
    _debug_hint("GOOGLE_SERVICE_ACCOUNT_FILE is empty in configuration.")
    pytest.skip(
        "GOOGLE_SERVICE_ACCOUNT_FILE must be configured for calendar integration tests",
        allow_module_level=True,
    )  # type: ignore[arg-type]

service_account_path = Path(service_account_file).expanduser()
if not service_account_path.exists():
    _debug_hint(f"Service account file not found at {service_account_path}.")
    pytest.skip(
        f"Google service account file not found at {service_account_path}",
        allow_module_level=True,
    )  # type: ignore[arg-type]

test_calendar_id = os.getenv("TEST_CALENDAR_ID") or os.getenv("CALENDAR_ID") or settings.calendar_id
if not test_calendar_id:
    _debug_hint(
        "Provide TEST_CALENDAR_ID or CALENDAR_ID environment variable, or set calendar_id in configuration for calendar integration tests."
    )
    pytest.skip(
        "Configure TEST_CALENDAR_ID / CALENDAR_ID environment variable (or calendar_id setting) to run calendar integration tests",
        allow_module_level=True,
    )  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def calendar_client() -> CalendarClient:
    return CalendarClient(
        service_account_file=str(service_account_path),
        default_timezone=settings.calendar_timezone or "UTC",
    )


def test_calendar_client_can_create_and_cancel_event(calendar_client: CalendarClient) -> None:
    start = datetime.now(tz=timezone.utc) + timedelta(minutes=5)
    end = start + timedelta(minutes=30)
    body = {
        "summary": "Integration Test Appointment",
        "description": "Calendar integration test event.",
        "start": {"dateTime": start.isoformat(), "timeZone": calendar_client.default_timezone},
        "end": {"dateTime": end.isoformat(), "timeZone": calendar_client.default_timezone},
    }
    created_event = calendar_client.create_event(calendar_id=test_calendar_id, body=body)
    event_id = created_event.get("id")
    assert event_id, "Expected Google Calendar to return an event id"

    try:
        cancelled = calendar_client.patch_event(
            calendar_id=test_calendar_id,
            event_id=event_id,
            body={"status": "cancelled"},
        )
        assert cancelled.get("status") == "cancelled"
    finally:
        if event_id:
            try:
                calendar_client.patch_event(
                    calendar_id=test_calendar_id,
                    event_id=event_id,
                    body={"status": "cancelled"},
                )
            except Exception:  # pragma: no cover - best effort cleanup
                pass
