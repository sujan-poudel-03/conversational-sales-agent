from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Callable, Dict

from langgraph.graph import END, StateGraph

from src.orchestrator.intents import Intent
from src.orchestrator.state import ConversationState
from src.services.calendar import CalendarService
from src.services.lead import LeadService
from src.services.rag import RagService


class AgentOrchestrator:
    """LangGraph-based state machine for the conversational sales agent."""

    def __init__(
        self,
        rag_service: RagService,
        lead_service: LeadService,
        calendar_service: CalendarService,
        intent_classifier: Callable[[ConversationState], Intent],
    ) -> None:
        self._rag_service = rag_service
        self._lead_service = lead_service
        self._calendar_service = calendar_service
        self._intent_classifier = intent_classifier
        self._graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph[ConversationState]:
        graph: StateGraph[ConversationState] = StateGraph(ConversationState)

        graph.add_node("intent_classifier", self._intent_node)
        graph.add_node("rag_chain", self._rag_node)
        graph.add_node("lead_capture", self._lead_node)
        graph.add_node("lead_saver", self._lead_saver_node)
        graph.add_node("booking", self._booking_node)

        graph.set_entry_point("intent_classifier")

        graph.add_conditional_edges(
            "intent_classifier",
            self._intent_router,
            {
                Intent.RAG_INFO: "rag_chain",
                Intent.PURCHASE_INTEREST: "lead_capture",
                Intent.BOOKING: "lead_capture",
                Intent.CANCEL_BOOKING: "booking",
            },
        )

        graph.add_edge("lead_capture", "lead_saver")
        graph.add_conditional_edges(
            "lead_saver",
            self._post_lead_router,
            {
                True: "booking",
                False: END,
            },
        )
        graph.add_edge("rag_chain", END)
        graph.add_edge("booking", END)

        return graph

    def _intent_node(self, state: ConversationState) -> ConversationState:
        updated = state.copy()
        updated.intent = self._intent_classifier(updated)
        return updated

    def _rag_node(self, state: ConversationState) -> ConversationState:
        updated = state.copy()
        response = self._rag_service.answer_query(
            context=updated.context, query=updated.user_query, history=updated.history
        )
        updated.history.append({"role": "assistant", "content": response})
        return updated

    def _lead_node(self, state: ConversationState) -> ConversationState:
        updated = state.copy()
        capture_result = self._lead_service.capture_lead_step(
            context=updated.context,
            user_query=updated.user_query,
            existing_lead=updated.lead_data,
        )
        updated.lead_data.update(capture_result.updates)

        if capture_result.prompt:
            updated.history.append({"role": "assistant", "content": capture_result.prompt})
        elif self._lead_service.is_complete(updated.lead_data):
            confirmation = self._lead_service.build_confirmation_message(updated.lead_data)
            updated.history.append({"role": "assistant", "content": confirmation})
        else:
            updated.history.append(
                {
                    "role": "assistant",
                    "content": "Thanks for the details - feel free to share more so I can complete your request.",
                }
            )
        return updated

    def _lead_saver_node(self, state: ConversationState) -> ConversationState:
        updated = state.copy()
        if self._lead_service.is_complete(updated.lead_data):
            lead_record = self._lead_service.persist_lead(updated.context, updated.lead_data)
            updated.history.append({"role": "system", "content": f"Lead saved: {lead_record['id']}"})
        return updated

    def _post_lead_router(self, state: ConversationState) -> bool:
        if state.intent == Intent.BOOKING and self._lead_service.is_complete(state.lead_data):
            return True
        return False

    def _booking_node(self, state: ConversationState) -> ConversationState:
        updated = state.copy()
        booking_result = self._calendar_service.handle_booking(
            context=updated.context,
            user_query=updated.user_query,
            lead_data=updated.lead_data,
            appointment_id=updated.appointment_id,
            intent=updated.intent,
        )
        updated.appointment_id = booking_result.appointment_id
        updated.history.append({"role": "assistant", "content": booking_result.message})
        if booking_result.audit_note:
            updated.history.append({"role": "system", "content": booking_result.audit_note})
        return updated

    def _intent_router(self, state: ConversationState) -> Intent:
        return state.intent

    def run(self, state: ConversationState) -> ConversationState:
        payload = asdict(state) if is_dataclass(state) else state
        result = self._graph.invoke(payload)
        if isinstance(result, ConversationState):
            return result
        if isinstance(result, dict):
            intent_value = result.get("intent", Intent.RAG_INFO)
            if not isinstance(intent_value, Intent):
                intent_value = Intent(intent_value)
            return ConversationState(
                intent=intent_value,
                user_query=result.get("user_query", ""),
                context=dict(result.get("context", {})),
                lead_data=dict(result.get("lead_data", {})),
                appointment_id=result.get("appointment_id"),
                history=list(result.get("history", [])),
            )
        raise TypeError(f"Unsupported state result from graph: {type(result)!r}")

    def lead_is_complete(self, lead_data: Dict[str, str]) -> bool:
        return self._lead_service.is_complete(lead_data)

