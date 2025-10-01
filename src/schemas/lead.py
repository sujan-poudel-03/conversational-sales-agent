from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, EmailStr

from src.schemas.context import TenantContext


class LeadCapturePayload(BaseModel):
    context: TenantContext
    name: Optional[str]
    email: Optional[EmailStr]
    phone: Optional[str]
    product_interest: List[str] = Field(default_factory=list)
    interest_reason: Optional[str]
    budget_expectation: Optional[str]


class LeadRecord(LeadCapturePayload):
    lead_status: str = Field(default="NEW")
    capture_timestamp: datetime = Field(default_factory=datetime.utcnow)