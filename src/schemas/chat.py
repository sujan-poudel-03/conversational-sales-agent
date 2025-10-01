from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from src.schemas.context import TenantContext


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role such as user, assistant, system")
    content: str = Field(..., description="Plain text content")


class ChatRequest(BaseModel):
    context: TenantContext
    message: ChatMessage
    history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    intent: str
    lead_captured: bool = False
    appointment_id: Optional[str] = None