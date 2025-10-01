# Conversational Sales Agent

This project implements a multi-tenant conversational sales agent that orchestrates retrieval augmented generation, lead capture, and appointment management using LangGraph.

## Project Structure

```
src/
  app/            # FastAPI application entrypoint & wiring
  orchestrator/   # LangGraph state machine and state definitions
  services/       # Business services for RAG, leads, calendar, intent
  adapters/       # Integrations with external systems (Pinecone, Mongo, etc.)
  ingestion/      # Knowledge base ingestion pipeline utilities
  schemas/        # Pydantic schemas for API payloads
  utils/          # Shared utilities (logging)
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set environment variables (or `.env` file) for Pinecone, MongoDB, Google, and email providers.
3. Run the API server:
   ```bash
   uvicorn src.app.main:app --reload
   ```

### Running Tests

```bash
pytest
```

## Key Flows

- **Chat** `/api/v1/chat` routes user messages through LangGraph to classify intent, invoke the RAG service, capture lead data, and manage bookings.
- **Ingestion** `/api/v1/ingest` upserts pre-chunked vectors into Pinecone with tenant metadata for strict isolation.
- **Appointments** The calendar service maps tenant context to Google Calendar IDs and oversees booking lifecycle, including cancellation.

Replace the heuristic intent classifier with `IntentClassifier` that uses Gemini when ready for production workloads.