from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from src.schemas.context import TenantContext


class IngestionDocument(BaseModel):
    text: Optional[str] = Field(default=None, description="Inline content to ingest.")
    source_path: Optional[str] = Field(default=None, description="Relative or absolute path to a file on disk.")

    @model_validator(mode='after')
    def _validate_source(cls, document: 'IngestionDocument') -> 'IngestionDocument':
        if not document.text and not document.source_path:
            raise ValueError('Either text or source_path must be provided for ingestion.')
        return document


class IngestionRequest(BaseModel):
    context: TenantContext
    documents: List[IngestionDocument] = Field(default_factory=list)


class IngestionStatus(BaseModel):
    processed: int
    failed: int
    message: str