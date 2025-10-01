from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.orchestrator.intents import Intent


@dataclass
class ConversationState:
    intent: Intent = Intent.RAG_INFO
    user_query: str = ""
    context: Dict[str, str] = field(default_factory=dict)
    lead_data: Dict[str, Optional[str]] = field(default_factory=dict)
    appointment_id: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)

    def copy(self) -> "ConversationState":
        return ConversationState(
            intent=self.intent,
            user_query=self.user_query,
            context=dict(self.context),
            lead_data=dict(self.lead_data),
            appointment_id=self.appointment_id,
            history=list(self.history),
        )