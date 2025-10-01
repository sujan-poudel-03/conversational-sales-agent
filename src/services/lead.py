from __future__ import annotations

from typing import Dict, Optional

from src.adapters.email_client import EmailClient


class LeadService:
    """Captures lead information and persists it to MongoDB."""

    REQUIRED_FIELDS = {"name", "email", "product_interest", "interest_reason"}

    def __init__(self, collection, email_client: EmailClient) -> None:
        self._collection = collection
        self._email_client = email_client

    def capture_lead_step(
        self,
        context: Dict[str, str],
        user_query: str,
        existing_lead: Dict[str, Optional[str]],
    ) -> Dict[str, Optional[str]]:
        # Placeholder: real implementation would delegate to LLM with LangGraph prompts
        lead_data = dict(existing_lead)
        if "@" in user_query and "email" not in lead_data:
            lead_data["email"] = user_query.strip()
        if "budget" in user_query.lower():
            lead_data["budget_expectation"] = user_query
        lead_data.setdefault("org_id", context["org_id"])
        lead_data.setdefault("branch_id", context["branch_id"])
        return lead_data

    def is_complete(self, lead_data: Dict[str, Optional[str]]) -> bool:
        return all(field in lead_data and lead_data[field] for field in self.REQUIRED_FIELDS)

    def persist_lead(self, context: Dict[str, str], lead_data: Dict[str, Optional[str]]):
        payload = {
            "org_id": context["org_id"],
            "branch_id": context["branch_id"],
            "name": lead_data.get("name"),
            "email": lead_data.get("email"),
            "phone": lead_data.get("phone"),
            "product_interest": lead_data.get("product_interest", []),
            "interest_reason": lead_data.get("interest_reason"),
            "budget_expectation": lead_data.get("budget_expectation"),
            "lead_status": lead_data.get("lead_status", "NEW"),
        }
        result = self._collection.insert_one(payload)
        lead_id = str(result.inserted_id)
        if payload["email"]:
            self._email_client.send(
                recipient=payload["email"],
                subject="Thanks for your interest",
                body="We will reach out shortly with more information.",
            )
        return {"id": lead_id, **payload}