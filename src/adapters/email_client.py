from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class EmailClient:
    api_key: str
    sender_domain: str

    def send(self, recipient: str, subject: str, body: str) -> Dict[str, str]:  # pragma: no cover
        # Placeholder for integration with SendGrid / SMTP provider
        if not self.api_key:
            raise RuntimeError("Email client configured without API key")
        return {"status": "queued", "recipient": recipient, "subject": subject}