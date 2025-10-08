from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration."""

    app_name: str = Field(default="Conversational Sales Agent")
    debug: bool = Field(default=False)

    # Vector DB
    pinecone_api_key: str = Field(default="")
    pinecone_environment: str = Field(default="")
    pinecone_index: str = Field(default="")
    pinecone_dimension: int = Field(default=1536)
    pinecone_metric: str = Field(default="cosine")
    pinecone_cloud: str = Field(default="")
    pinecone_region: str = Field(default="")
    pinecone_pod_type: str = Field(default="")

    # MongoDB
    mongo_uri: str = Field(default="mongodb://localhost:27017")
    mongo_database: str = Field(default="sales_agent")
    leads_collection: str = Field(default="leads")

    # Google Calendar
    google_service_account_file: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_SA_FILE", "GOOGLE_SERVICE_ACCOUNT_FILE"),
    )
    calendar_timezone: str = Field(default="UTC")

    # Email
    email_sender_domain: str = Field(default="")
    email_api_key: str = Field(default="")

    # LangGraph / LLM
    gemini_api_key: str = Field(default="")
    gemini_embedding_model: str = Field(
        default="text-embedding-004",
        validation_alias=AliasChoices("GEMINI_EMBED_MODEL", "GEMINI_EMBEDDING_MODEL"),
    )
    gemini_intent_model: str = Field(
        default="gemini-1.5-pro",
        validation_alias=AliasChoices("GEMINI_INTENT_MODEL"),
    )
    allowed_origins: List[str] = Field(
        default_factory=list,
        validation_alias="ALLOWED_ORIGINS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
