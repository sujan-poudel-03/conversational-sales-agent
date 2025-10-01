from __future__ import annotations

from fastapi import APIRouter, Depends, status

from src.app.config import Settings
from src.app.dependencies import get_ingestion_pipeline, get_orchestrator, get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.orchestrator.intents import Intent
from src.orchestrator.state import ConversationState
from src.schemas.chat import ChatRequest, ChatResponse
from src.schemas.ingestion import IngestionRequest, IngestionStatus

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
def health(settings: Settings = Depends(get_settings)) -> dict:
    return {"app": settings.app_name, "status": "ok"}


@router.post("/api/v1/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    orchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    context_dict = payload.context.dict()
    state = ConversationState(
        intent=Intent.RAG_INFO,
        user_query=payload.message.content,
        context=context_dict,
        history=[message.dict() for message in payload.history],
    )
    final_state = orchestrator.run(state)
    reply = final_state.history[-1]["content"] if final_state.history else ""
    return ChatResponse(
        reply=reply,
        intent=final_state.intent.value,
        lead_captured=orchestrator.lead_is_complete(final_state.lead_data),
        appointment_id=final_state.appointment_id,
    )


@router.post("/api/v1/ingest", response_model=IngestionStatus)
def ingest(
    payload: IngestionRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestionStatus:
    status_payload = pipeline.run(
        context=payload.context.dict(),
        documents=[doc.dict() for doc in payload.documents],
    )
    return IngestionStatus(
        processed=status_payload.get("processed", 0),
        failed=status_payload.get("failed", 0),
        message="Ingestion completed",
    )