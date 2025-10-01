from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
except ImportError:  # pragma: no cover
    service_account = None
    build = None


@dataclass
class CalendarClient:
    service_account_file: str
    default_timezone: str
    scopes: tuple[str, ...] = ("https://www.googleapis.com/auth/calendar",)

    def _service(self):
        if not service_account or not build:
            raise RuntimeError("google-api-python-client is required for Calendar access")
        credentials = service_account.Credentials.from_service_account_file(
            self.service_account_file, scopes=self.scopes
        )
        delegated = credentials.with_subject(credentials.service_account_email)
        return build("calendar", "v3", credentials=delegated, cache_discovery=False)

    def list_events(self, calendar_id: str, time_min: str, time_max: str) -> Dict[str, Any]:
        service = self._service()
        events = (
            service.events()
            .list(calendarId=calendar_id, timeMin=time_min, timeMax=time_max, singleEvents=True)
            .execute()
        )
        return events

    def create_event(self, calendar_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        service = self._service()
        event = service.events().insert(calendarId=calendar_id, body=body).execute()
        return event

    def patch_event(self, calendar_id: str, event_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        service = self._service()
        event = service.events().patch(calendarId=calendar_id, eventId=event_id, body=body).execute()
        return event