# Task Requirements Checklist

## 1. Multi-Tenant RAG (Knowledge Base Access)
- [x] Connect to Pinecone (vector DB)
  - Verified via ingestion pipeline and live integration test (`tests/test_ingest_api.py`).
- [x] Ensure KB selection via `org_id` + `branch_id`
  - Namespacing enforced in ingestion pipeline (`org::branch`) and exercised in integration tests.
- [ ] Add automated coverage for cross-tenant isolation for retrieval (follow-up)

## 2. Conversational Lead Capture
- [ ] Detect purchase intent ahead of KB answer (confirm runtime behaviour and tests)
- [ ] Capture structured lead details conversationally
- [ ] Persist captured leads to MongoDB
- [ ] Add tests covering lead capture happy-path and edge cases

## 3. Appointment Management
- [ ] Support booking/reschedule/cancel flows end-to-end
- [ ] Sync bookings with Google Calendar per org/branch
- [ ] Send confirmation email after booking updates
- [ ] Backfill tests for calendar/email paths

## 4. Architecture & Orchestration (LangGraph Flow)
- [x] Structure conversation flow scaffold (intent classifier + orchestrator skeleton)
  - Rule-based classifier and orchestrator wiring exist; needs validation with new integrations.
- [ ] Fully implement lead capture branch within graph
- [ ] Integrate Pinecone RAG response node with updated embeddings
- [ ] Implement booking workflow nodes (Calendar + Email)
- [ ] Create end-to-end tests covering main intents
