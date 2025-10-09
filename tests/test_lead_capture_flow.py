from __future__ import annotations

from types import SimpleNamespace

from src.services.lead import LeadService


class MemoryCollection:
    def __init__(self) -> None:
        self.inserted = []

    def insert_one(self, payload):
        self.inserted.append(payload)
        return SimpleNamespace(inserted_id="lead-001")


class MemoryEmailClient:
    def __init__(self) -> None:
        self.sent = []

    def send(self, recipient: str, subject: str, body: str):
        self.sent.append({"recipient": recipient, "subject": subject, "body": body})
        return {"status": "queued"}


def test_lead_capture_conversation_and_persistence():
    service = LeadService(collection=MemoryCollection(), email_client=MemoryEmailClient())
    context = {"org_id": "org-alpha", "branch_id": "branch-west"}
    lead_state = {}

    inputs = [
        "Hello, I'm Casey Lee and I'm interested in heat pumps because winters are rough.",
        "You can reach me at casey.lee@example.com.",
        "My number is +1 555 111 2222 and our budget is around $7500.",
    ]

    prompts = []
    for utterance in inputs:
        result = service.capture_lead_step(context, utterance, lead_state)
        lead_state.update(result.updates)
        prompts.append(result.prompt)

    assert service.is_complete(lead_state) is True
    assert lead_state["name"] == "Casey Lee"
    assert lead_state["email"] == "casey.lee@example.com"
    assert lead_state["phone"] == "+1 555 111 2222"
    assert lead_state["interest_reason"].lower().startswith("winters are rough")
    assert lead_state["product_interest"][0].startswith("heat pumps")
    assert lead_state["budget_expectation"] == "$7500"
    assert prompts[-1] is None

    record = service.persist_lead(context, lead_state)
    assert record["id"] == "lead-001"
    assert service._collection.inserted  # type: ignore[attr-defined]
    assert service._email_client.sent  # type: ignore[attr-defined]
