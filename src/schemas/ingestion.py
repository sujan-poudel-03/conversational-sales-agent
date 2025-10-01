from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from src.schemas.context import TenantContext


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    source_file: Optional[str]
    sequence: int


class IngestionRequest(BaseModel):
    context: TenantContext
    documents: List[DocumentChunk] = Field(default_factory=list)


class IngestionStatus(BaseModel):
    processed: int
    failed: int
    message: str