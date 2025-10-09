from __future__ import annotations

import os

import pytest

from src.app.config import get_settings
from src.app.dependencies import get_email_client

settings = get_settings()

if not settings.email_api_key or not settings.email_sender_email:
    print(
        "[email-test] Missing EMAIL_API_KEY or EMAIL_SENDER_EMAIL configuration. "
        f"email_api_key set={bool(settings.email_api_key)} "
        f"email_sender_email set={bool(settings.email_sender_email)}"
    )
    pytest.skip(
        "EMAIL_API_KEY and EMAIL_SENDER_EMAIL (or EMAIL_SENDER_DOMAIN) must be configured for email service tests",
        allow_module_level=True,
    )  # type: ignore[arg-type]

test_recipient = os.getenv("EMAIL_TEST_RECIPIENT") or "poudelsujan03@gmail.com"

if not test_recipient:
    print("[email-test] EMAIL_TEST_RECIPIENT environment variable is not set.")
    pytest.skip(
        "Set EMAIL_TEST_RECIPIENT environment variable to run the email integration test",
        allow_module_level=True,
    )  # type: ignore[arg-type]


def test_email_client_send_returns_status() -> None:
    client = get_email_client()
    result = client.send(
        recipient=test_recipient,
        subject="Conversational Sales Agent integration check",
        body="Testing outbound email from the MVP flow.",
    )
    assert result.get("status") in (200, 202)
