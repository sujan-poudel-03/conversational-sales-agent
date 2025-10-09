from __future__ import annotations

from pydantic import BaseModel, Field


class TenantContext(BaseModel):
    org_id: str = Field(..., description="Tenant (organization) identifier")
    branch_id: str = Field(..., description="Branch/location identifier within the tenant")
    user_session_id: str = Field(..., description="Conversation session identifier")
    calendar_id: str | None = Field(
        default=None,
        description="Optional calendar identifier used for booking operations",
    )
