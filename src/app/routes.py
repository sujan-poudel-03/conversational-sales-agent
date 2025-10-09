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


def _extract_user_reply(history: list[dict[str, str]]) -> str:
    """Return the last user-facing message, skipping internal audit entries."""
    for message in reversed(history):
        content = message.get("content", "")
        if not content:
            continue
        if message.get("role") == "system" and (
            content.startswith("Lead saved") or content.startswith("calendar_event_")
        ):
            continue
        return content
    return ""


@router.get("/health", status_code=status.HTTP_200_OK)
def health(settings: Settings = Depends(get_settings)) -> dict:
    return {"app": settings.app_name, "status": "ok"}


@router.post("/api/v1/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    orchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    context_dict = payload.context.model_dump()
    state = ConversationState(
        intent=Intent.RAG_INFO,
        user_query=payload.message.content,
        context=context_dict,
        history=[message.model_dump() for message in payload.history],
    )
    final_state = orchestrator.run(state)
    print("Final conversation state:", final_state)
    reply = _extract_user_reply(final_state.history)
    print("Final reply:", reply)
    return ChatResponse(
        reply=reply,
        intent=final_state.intent.value.lower(),
        lead_captured=orchestrator.lead_is_complete(final_state.lead_data),
        appointment_id=final_state.appointment_id,
    )


@router.post("/api/v1/ingest", response_model=IngestionStatus)
def ingest(
    payload: IngestionRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestionStatus:
    status_payload = pipeline.run(
        context=payload.context.model_dump(),
        documents=[doc.model_dump() for doc in payload.documents],
    )
    return IngestionStatus(
        processed=status_payload.get("processed", 0),
        failed=status_payload.get("failed", 0),
        message="Ingestion completed",
    )
