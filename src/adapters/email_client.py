from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:  # pragma: no cover - import guard for optional dependency
    import requests
except ImportError:  # pragma: no cover - handled at runtime
    requests = None  # type: ignore


SENDGRID_ENDPOINT = "https://api.sendgrid.com/v3/mail/send"


@dataclass
class EmailClient:
    api_key: str
    sender_email: str

    def send(self, recipient: str, subject: str, body: str) -> Dict[str, str]:
        if not self.api_key:
            raise RuntimeError("Email client configured without API key")
        if not self.sender_email:
            raise RuntimeError("Email client configured without sender email")
        if requests is None:
            raise RuntimeError("The 'requests' package is required for SendGrid email sending")

        payload = {
            "personalizations": [
                {
                    "to": [{"email": recipient}],
                }
            ],
            "from": {"email": self.sender_email},
            "subject": subject,
            "content": [
                {
                    "type": "text/plain",
                    "value": body,
                }
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(SENDGRID_ENDPOINT, headers=headers, json=payload, timeout=10)
        if response.status_code not in (200, 202):
            raise RuntimeError(
                f"Failed to send email via SendGrid (status {response.status_code}): {response.text}"
            )

        return {
            "status": response.status_code,
            "recipient": recipient,
            "subject": subject,
        }
