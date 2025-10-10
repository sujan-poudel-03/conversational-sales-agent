from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional

from src.adapters.email_client import EmailClient


@dataclass
class LeadCaptureResult:
    updates: Dict[str, Optional[str]]
    prompt: Optional[str] = None
    completed: bool = False


class LeadService:
    """Captures lead information and persists it to MongoDB."""

    REQUIRED_FIELDS = ["product_interest", "name", "email"]
    OPTIONAL_FIELDS = ["interest_reason", "budget_expectation", "phone"]
    PROMPTS = {
        "product_interest": "I can help with that. Which products are you most interested in?",
        "interest_reason": "Thanks! What makes this a good fit for you right now?",
        "name": "Great - could you share your name so we know who to contact?",
        "email": "What's the best email to reach you at?",
        "budget_expectation": "Do you have a budget or price range in mind?",
        "phone": "If you'd like, share a phone number so our team can text or call you.",
    }

    PRODUCT_PATTERNS = [
        re.compile(r"interested in ([^.?!]+)", re.IGNORECASE),
        re.compile(r"looking for ([^.?!]+)", re.IGNORECASE),
        re.compile(r"need ([^.?!]+)", re.IGNORECASE),
        re.compile(r"want ([^.?!]+)", re.IGNORECASE),
    ]
    REASON_PATTERNS = [
        re.compile(r"because ([^.?!]+)", re.IGNORECASE),
        re.compile(r"since ([^.?!]+)", re.IGNORECASE),
        re.compile(r"so that ([^.?!]+)", re.IGNORECASE),
        re.compile(r"as ([^.?!]+)", re.IGNORECASE),
        re.compile(r"to ([^.?!]+)", re.IGNORECASE),
    ]
    NAME_PATTERNS = [
        re.compile(r"my name is\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*)", re.IGNORECASE),
        re.compile(r"I'm\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"),
        re.compile(r"I am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"),
        re.compile(r"(?:Mr|Mrs|Ms|Dr)\.?\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*)", re.IGNORECASE),
    ]

    def __init__(self, collection, email_client: EmailClient) -> None:
        self._collection = collection
        self._email_client = email_client

    def capture_lead_step(
        self,
        context: Dict[str, str],
        user_query: str,
        existing_lead: Dict[str, Optional[str]],
    ) -> LeadCaptureResult:
        lead_data = dict(existing_lead)
        updates: Dict[str, Optional[str]] = {}

        def set_field(field: str, value: Optional[str]):
            if not value:
                return
            if field == "product_interest":
                existing_value = lead_data.get(field) or []
                if isinstance(existing_value, str):
                    existing_list = [existing_value]
                else:
                    existing_list = list(existing_value)
                values = [item.strip() for item in re.split(r",| and ", value) if item.strip()]
                merged = list(dict.fromkeys([*existing_list, *values]))
                if merged and merged != existing_list:
                    lead_data[field] = merged
                    updates[field] = merged
            else:
                if not lead_data.get(field):
                    lead_data[field] = value.strip()
                    updates[field] = value.strip()

        text = user_query.strip()

        email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        set_field("email", email_match.group(0) if email_match else None)

        phone_match = re.search(r"(\+?\d[\d\s\-]{7,}\d)", text)
        set_field("phone", phone_match.group(0) if phone_match else None)

        name_match = self._extract_by_patterns(text, self.NAME_PATTERNS)
        set_field("name", name_match)

        product_match = self._extract_by_patterns(text, self.PRODUCT_PATTERNS)
        if product_match:
            product_match = self._trim_after_reason(product_match)
        set_field("product_interest", product_match)

        reason_match = self._extract_by_patterns(text, self.REASON_PATTERNS)
        if not reason_match:
            reason_match = self._infer_reason_from_intent(text)
        set_field("interest_reason", reason_match)

        if not lead_data.get("product_interest"):
            fallback_product = self._infer_product_from_short_reply(text)
            set_field("product_interest", fallback_product)

        budget_match = re.search(r"(?:budget|around|about|roughly)\s*(?:is|:)?\s*(\$?\d[\d,]*(?:\.\d{1,2})?)", text, re.IGNORECASE)
        set_field("budget_expectation", budget_match.group(1) if budget_match else None)

        lead_data.setdefault("org_id", context.get("org_id"))
        lead_data.setdefault("branch_id", context.get("branch_id"))

        missing_required = [field for field in self.REQUIRED_FIELDS if not lead_data.get(field)]
        next_field = missing_required[0] if missing_required else None
        prompt = self.PROMPTS.get(next_field) if next_field else None

        completed = not missing_required
        return LeadCaptureResult(updates=updates, prompt=prompt, completed=completed)

    def _extract_by_patterns(self, text: str, patterns: List[re.Pattern]) -> Optional[str]:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return None

    def _trim_after_reason(self, text: str) -> str:
        lowered = text.lower()
        for token in (' because ', ' since ', ' so that '):
            marker = lowered.find(token)
            if marker != -1:
                return text[:marker].strip()
        return text.strip()

    def _infer_product_from_short_reply(self, text: str) -> Optional[str]:
        normalized = text.strip()
        if not normalized:
            return None
        lower = normalized.lower()
        if lower in {
            "yes",
            "yeah",
            "yep",
            "no",
            "nope",
            "ok",
            "okay",
            "sure",
            "maybe",
            "not sure",
            "thanks",
            "thank you",
        }:
            return None
        if "?" in normalized or "@" in normalized or normalized.replace(".", "").isdigit():
            return None
        words = normalized.split()
        if len(words) > 12:
            return None
        prefixes = ["i am ", "i'm ", "we are ", "we're ", "we need ", "i need ", "i want "]
        for prefix in prefixes:
            if lower.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                lower = normalized.lower()
                break
        normalized = normalized.rstrip(".!")
        if not normalized:
            return None
        return normalized

    def _infer_reason_from_intent(self, text: str) -> Optional[str]:
        lowered = text.lower()
        tokens = [" to ", " so we can ", " so i can "]
        for token in tokens:
            idx = lowered.find(token)
            if idx != -1:
                return text[idx + len(token):].strip()
        return None

    def is_complete(self, lead_data: Dict[str, Optional[str]]) -> bool:
        for field in self.REQUIRED_FIELDS:
            value = lead_data.get(field)
            if not value:
                return False
        return True

    def build_confirmation_message(self, lead_data: Dict[str, Optional[str]]) -> str:
        pieces = []
        products = lead_data.get("product_interest")
        if products:
            if isinstance(products, (list, tuple)):
                product_text = ", ".join(products)
            else:
                product_text = str(products)
            pieces.append(f"You're interested in {product_text}.")
        reason = lead_data.get("interest_reason")
        if reason:
            pieces.append(f"Reason noted: {reason}.")
        budget = lead_data.get("budget_expectation")
        if budget:
            pieces.append(f"Budget around {budget}.")
        contact_bits = []
        if lead_data.get("name"):
            contact_bits.append(lead_data["name"])
        if lead_data.get("email"):
            contact_bits.append(lead_data["email"])
        if lead_data.get("phone"):
            contact_bits.append(lead_data["phone"])
        if contact_bits:
            pieces.append("Contact details: " + ", ".join(contact_bits) + ".")
        pieces.append("I'll pass this along to our sales team - anything else you'd like to add?")
        return " ".join(pieces)

    def persist_lead(self, context: Dict[str, str], lead_data: Dict[str, Optional[str]]):
        product_interest = lead_data.get("product_interest")
        if isinstance(product_interest, str):
            product_interest_value: List[str] = [product_interest]
        elif isinstance(product_interest, list):
            product_interest_value = product_interest
        elif product_interest is None:
            product_interest_value = []
        else:
            product_interest_value = list(product_interest)

        payload = {
            "org_id": context["org_id"],
            "branch_id": context["branch_id"],
            "name": lead_data.get("name"),
            "email": lead_data.get("email"),
            "phone": lead_data.get("phone"),
            "product_interest": product_interest_value,
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
