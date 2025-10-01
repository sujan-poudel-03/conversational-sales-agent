from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from src.app.config import Settings, get_settings
from src.adapters.calendar_client import CalendarClient
from src.adapters.email_client import EmailClient
from src.adapters.mongo_client import MongoClientFactory
from src.adapters.pinecone_client import PineconeClientFactory
from src.ingestion.pipeline import IngestionPipeline
from src.orchestrator.graph import AgentOrchestrator
from src.services.calendar import CalendarService
from src.services.embeddings_fallback import DeterministicEmbedding
from src.services.intent_rules import RuleBasedIntentClassifier
from src.services.lead import LeadService
from src.services.rag import RagService


@lru_cache(maxsize=1)
def get_mongo_factory(settings: Settings = Depends(get_settings)) -> MongoClientFactory:
    return MongoClientFactory(settings.mongo_uri, settings.mongo_database)


@lru_cache(maxsize=1)
def get_pinecone_factory(settings: Settings = Depends(get_settings)) -> PineconeClientFactory:
    return PineconeClientFactory(
        api_key=settings.pinecone_api_key,
        environment=settings.pinecone_environment,
        index_name=settings.pinecone_index,
    )


@lru_cache(maxsize=1)
def get_email_client(settings: Settings = Depends(get_settings)) -> EmailClient:
    return EmailClient(api_key=settings.email_api_key, sender_domain=settings.email_sender_domain)


@lru_cache(maxsize=1)
def get_calendar_client(settings: Settings = Depends(get_settings)) -> CalendarClient:
    return CalendarClient(
        service_account_file=settings.google_service_account_file,
        default_timezone=settings.calendar_timezone,
    )


@lru_cache(maxsize=1)
def get_embedder(settings: Settings = Depends(get_settings)):
    if settings.gemini_api_key:
        try:
            from src.services.embeddings import EmbeddingService

            return EmbeddingService(settings.gemini_embedding_model, settings.gemini_api_key)
        except RuntimeError:
            pass
    return DeterministicEmbedding()


def get_lead_service(
    settings: Settings = Depends(get_settings),
    mongo_factory: MongoClientFactory = Depends(get_mongo_factory),
    email_client: EmailClient = Depends(get_email_client),
) -> LeadService:
    return LeadService(
        collection=mongo_factory.get_collection(settings.leads_collection),
        email_client=email_client,
    )


def get_rag_service(
    pinecone_factory: PineconeClientFactory = Depends(get_pinecone_factory),
    embedder = Depends(get_embedder),
) -> RagService:
    return RagService(pinecone_index=pinecone_factory.get_index(), embedder=embedder)


def get_ingestion_pipeline(
    pinecone_factory: PineconeClientFactory = Depends(get_pinecone_factory),
) -> IngestionPipeline:
    return IngestionPipeline(pinecone_index=pinecone_factory.get_index())


def get_calendar_service(
    calendar_client: CalendarClient = Depends(get_calendar_client),
) -> CalendarService:
    return CalendarService(calendar_client=calendar_client)


def get_orchestrator(
    rag_service: RagService = Depends(get_rag_service),
    lead_service: LeadService = Depends(get_lead_service),
    calendar_service: CalendarService = Depends(get_calendar_service),
) -> AgentOrchestrator:
    classifier = RuleBasedIntentClassifier()
    return AgentOrchestrator(
        rag_service=rag_service,
        lead_service=lead_service,
        calendar_service=calendar_service,
        intent_classifier=classifier.classify,
    )