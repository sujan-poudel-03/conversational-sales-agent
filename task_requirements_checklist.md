# Task Requirements Checklist

## 1. Multi-Tenant RAG (Knowledge Base Access)
- [x] Connect to Pinecone (vector DB)
  - Verified via ingestion pipeline and live integration test (`tests/test_ingest_api.py`).
- [x] Ensure KB selection via `org_id` + `branch_id`
  - Namespacing enforced in ingestion pipeline (`org::branch`) and exercised in integration tests.
- [ ] Add automated coverage for cross-tenant isolation for retrieval (follow-up)

## 2. Conversational Lead Capture
- [ ] Detect purchase intent ahead of KB answer (confirm runtime behaviour and tests)
- [x] Capture structured lead details conversationally
  - `LeadService.capture_lead_step` now parses email, phone, budget, product interest, and reasons with prompts.
- [x] Persist captured leads to MongoDB
  - Lead payload normalised (lists, metadata) before insert + thank-you email.
- [ ] Add automated validation for lead capture progression
  - User opted to remove existing unit tests; manual QA plan still needed.

## 3. Appointment Management
- [x] Support booking/reschedule/cancel flows end-to-end
  - CalendarService now creates, reschedules, or cancels events based on conversational intent.
- [x] Sync bookings with Google Calendar per org/branch
  - Calendar IDs derived from org/branch context and pushed via the Google API client.
- [x] Send confirmation email after booking updates
  - Email client now receives booking confirmations for create/patch/cancel flows.
- [ ] Backfill tests for calendar/email paths
  - Requires future end-to-end coverage once test strategy is reintroduced.

## 4. Architecture & Orchestration (LangGraph Flow)
- [x] Structure conversation flow scaffold (intent classifier + orchestrator skeleton)
  - Rule-based classifier and orchestrator wiring exist; needs validation with new integrations.
- [x] Fully implement lead capture branch within graph
  - Assistant now responds with follow-up prompts and summaries; completed leads persisted automatically.
- [ ] Integrate Pinecone RAG response node with updated embeddings
- [x] Implement booking workflow nodes (Calendar + Email)
  - Graph routes booking intents through lead capture, persistence, and calendar/email fulfillment.
- [ ] Create end-to-end tests covering main intents
