from __future__ import annotations
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app.config import get_settings
from src.app.main import app

try:  # ensure runtime deps for full integration are present
    import pymongo  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - env guard
    pytest.skip("pymongo is required for chat API integration tests", allow_module_level=True)  # type: ignore[arg-type]

try:
    from google.oauth2 import service_account  # type: ignore  # noqa: F401
    from googleapiclient.discovery import build  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - env guard
    pytest.skip("google-api-python-client is required for Calendar integration", allow_module_level=True)  # type: ignore[arg-type]

try:
    from google import genai  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - env guard
    pytest.skip("google-genai is required for RAG integration", allow_module_level=True)  # type: ignore[arg-type]


settings = get_settings()
required_settings = {
    "PINECONE_API_KEY": settings.pinecone_api_key,
    "PINECONE_INDEX": settings.pinecone_index,
    "GEMINI_API_KEY": settings.gemini_api_key,
    "MONGO_URI": settings.mongo_uri,
    "EMAIL_API_KEY": settings.email_api_key,
    "GOOGLE_SERVICE_ACCOUNT_FILE": settings.google_service_account_file,
}
missing_settings = [name for name, value in required_settings.items() if not value]
if missing_settings:
    pytest.skip(
        f"Set the following settings for chat API integration test: {', '.join(missing_settings)}",
        allow_module_level=True,
    )  # type: ignore[arg-type]

# Static tenant identifiers for integration testing.
TEST_ORG_ID = "integration_org_demo"
TEST_BRANCH_ID = "integration_branch_demo"
TEST_CALENDAR_ID = settings.calendar_id
if not TEST_CALENDAR_ID:
    pytest.skip(
        "Configure calendar_id (or CALENDAR_ID/TEST_CALENDAR_ID env var) for chat API integration tests",
        allow_module_level=True,
    )  # type: ignore[arg-type]

SAMPLE_DOCUMENT = Path("tests/data/sample_document.txt").resolve()
assert SAMPLE_DOCUMENT.exists(), "Sample document for ingestion is missing"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def tenant_context(client: TestClient):
    branch_id = TEST_BRANCH_ID
    org_id = TEST_ORG_ID
    context = {
        "org_id": org_id,
        "branch_id": branch_id,
        "calendar_id": TEST_CALENDAR_ID,
        "user_session_id": f"session-{uuid.uuid4().hex}",
    }
    ingest_payload = {
        "context": context,
        "documents": [{"source_path": str(SAMPLE_DOCUMENT)}],
    }
    response = client.post("/api/v1/ingest", json=ingest_payload)
    assert response.status_code == 200, response.text
    ingest_result = response.json()
    assert ingest_result["failed"] == 0
    assert ingest_result["processed"] >= 1
    return {"org_id": org_id, "branch_id": branch_id, "calendar_id": TEST_CALENDAR_ID}


def _base_payload(context: dict[str, str], content: str) -> dict:
    return {
        "context": {
            "org_id": context["org_id"],
            "branch_id": context["branch_id"],
            "calendar_id": context["calendar_id"],
            "user_session_id": f"session-{uuid.uuid4().hex}",
        },
        "message": {"role": "user", "content": content},
        "history": [],
    }


def test_chat_rag_info_flow(client: TestClient, tenant_context: dict[str, str]):
    payload = _base_payload(tenant_context, "What knowledge base details can you share?")
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["intent"] == "rag_info"
    assert isinstance(data["reply"], str) and data["reply"].strip()


def test_chat_purchase_lead_flow(client: TestClient, tenant_context: dict[str, str]):
    payload = _base_payload(
        tenant_context,
        "I'm interested in solar panels for my home. Email me at jordan@example.com.",
    )
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["intent"] == "purchase_interest"
    assert data["lead_captured"] is False
    assert isinstance(data["reply"], str) and data["reply"].strip()


def test_chat_booking_flow(client: TestClient, tenant_context: dict[str, str]):
    payload = _base_payload(
        tenant_context,
        (
            "Please book an appointment for tomorrow. My name is Alex Rivera, "
            "email alex@example.com, phone +1 555 333 4444, interested in solar panels because of high bills."
        ),
    )
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["intent"] == "booking"
    assert data["lead_captured"] is True
    assert isinstance(data["appointment_id"], str) and data["appointment_id"]
    assert isinstance(data["reply"], str) and data["reply"].strip()
