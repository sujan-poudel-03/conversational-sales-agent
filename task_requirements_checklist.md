# Task Requirements Checklist

## 1. Multi-Tenant RAG (Knowledge Base Access)
- [x] Connect to Pinecone (vector DB)
  - Verified via ingestion pipeline and live integration test (`tests/test_ingest_api.py`).
- [x] Ensure KB selection via `org_id` + `branch_id`
  - Namespacing enforced in ingestion pipeline (`org::branch`) and exercised in integration tests.
- [x] Add automated coverage for cross-tenant isolation for retrieval
  - `tests/test_rag_service.py` asserts tenant filtering on the Pinecone index.

## 2. Conversational Lead Capture
- [x] Detect purchase intent ahead of KB answer
  - Semantic TF-IDF classifier replaces keyword heuristics (`src/services/intent_rules.py`) with unit coverage in `tests/test_intent_classifier.py`.
- [x] Capture structured lead details conversationally
  - `LeadService.capture_lead_step` now parses email, phone, budget, product interest, and reasons with prompts.
- [x] Persist captured leads to MongoDB
  - Lead payload normalised (lists, metadata) before insert + thank-you email.
- [x] Add automated validation for lead capture progression
  - Flow covered through `tests/test_lead_capture_flow.py` and orchestrator assertions in `tests/test_orchestrator_flow.py`.

## 3. Appointment Management
- [x] Support booking/reschedule/cancel flows end-to-end
  - CalendarService now creates, reschedules, or cancels events based on conversational intent.
- [x] Sync bookings with Google Calendar per org/branch
  - Calendar IDs derived from org/branch context and pushed via the Google API client.
- [x] Send confirmation email after booking updates
  - Email client now receives booking confirmations for create/patch/cancel flows.
- [x] Backfill tests for calendar/email paths
  - Unit coverage in `tests/test_calendar_service.py` plus integration flow in `tests/test_chat_api.py`.

## 4. Architecture & Orchestration (LangGraph Flow)
- [x] Structure conversation flow scaffold (intent classifier + orchestrator skeleton)
  - LangGraph orchestrator wired to semantic intent classifier and exercised via chat API integration.
- [x] Fully implement lead capture branch within graph
  - Assistant now responds with follow-up prompts and summaries; completed leads persisted automatically.
- [x] Integrate Pinecone RAG response node with updated embeddings
  - Gemini embedding service powers Pinecone queries; validated by `tests/test_chat_api.py` and `tests/test_rag_service.py`.
- [x] Implement booking workflow nodes (Calendar + Email)
  - Graph routes booking intents through lead capture, persistence, and calendar/email fulfillment.
- [x] Create end-to-end tests covering main intents
  - `tests/test_chat_api.py` exercises RAG info, purchase intent, and booking pathways.
