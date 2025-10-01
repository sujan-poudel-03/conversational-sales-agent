from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    RAG_INFO = "RAG_INFO"
    PURCHASE_INTEREST = "PURCHASE_INTEREST"
    BOOKING = "BOOKING"
    CANCEL_BOOKING = "CANCEL_BOOKING"

    @classmethod
    def from_label(cls, label: str) -> "Intent":
        try:
            return cls(label)
        except ValueError as exc:
            raise ValueError(f"Unsupported intent label: {label}") from exc