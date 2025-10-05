from types import SimpleNamespace

import pytest

from src.services.lead import LeadService


class FakeCollection:
    def __init__(self) -> None:
        self.inserted = []

    def insert_one(self, payload):
        self.inserted.append(payload)
        return SimpleNamespace(inserted_id="lead-123")


class FakeEmailClient:
    def __init__(self) -> None:
        self.sent = []

    def send(self, recipient: str, subject: str, body: str) -> None:
        self.sent.append({"recipient": recipient, "subject": subject, "body": body})


@pytest.fixture()
def lead_service():
    return LeadService(collection=FakeCollection(), email_client=FakeEmailClient())


def test_lead_capture_progression_and_persistence(lead_service):
    context = {"org_id": "org-1", "branch_id": "branch-9"}
    lead_data = {}

    messages = [
        "Hi there! I'm interested in solar panels for a new cafe because I want to cut energy costs.",
        "My name is Jordan Smith.",
        "You can reach me at jordan@example.com or +1 555 222 3333.",
        "Our budget is around $5,000 for the first phase.",
    ]

    prompts = []
    for message in messages:
        result = lead_service.capture_lead_step(context, message, lead_data)
        lead_data.update(result.updates)
        prompts.append(result.prompt)

    assert lead_service.is_complete(lead_data) is True
    assert lead_data["product_interest"] == ["solar panels for a new cafe"]
    assert lead_data["interest_reason"] == "I want to cut energy costs"
    assert lead_data["name"] == "Jordan Smith"
    assert lead_data["email"] == "jordan@example.com"
    assert lead_data["budget_expectation"] == "$5,000"
    assert lead_data["phone"] == "+1 555 222 3333"

    assert prompts[0] == LeadService.PROMPTS["name"]
    assert prompts[1] == LeadService.PROMPTS["email"]
    assert prompts[2] == LeadService.PROMPTS["budget_expectation"]
    assert prompts[3] is None

    record = lead_service.persist_lead(context, lead_data)
    assert record["id"] == "lead-123"
    assert record["product_interest"] == ["solar panels for a new cafe"]
    assert lead_service._collection.inserted  # type: ignore[attr-defined]
    assert lead_service._email_client.sent  # type: ignore[attr-defined]
