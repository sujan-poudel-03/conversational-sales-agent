from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import pytest

try:  # Ensure required runtime deps are available.
    import pydantic_settings  # type: ignore  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - env guard
    pytest.skip("pydantic-settings is required for this integration test", allow_module_level=True)  # type: ignore[arg-type]

try:
    import langgraph  # type: ignore  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - env guard
    pytest.skip("langgraph is required for this integration test", allow_module_level=True)  # type: ignore[arg-type]

import pinecone  # type: ignore

from pinecone.exceptions import NotFoundException  # type: ignore[attr-defined]

from src.app.config import get_settings
from src.app.main import app
from src.services.embeddings_fallback import DeterministicEmbedding


try:  # Pinecone >=3 provides the Pinecone client class.
    from pinecone import Pinecone as PineconeClient  # type: ignore
    try:
        from pinecone import ServerlessSpec  # type: ignore
    except ImportError:  # pragma: no cover - optional helper
        ServerlessSpec = None  # type: ignore
except ImportError:  # pragma: no cover - older SDK fallback
    PineconeClient = None  # type: ignore
    ServerlessSpec = None  # type: ignore


def _namespace_vector_count(index, namespace: str) -> int:
    stats = index.describe_index_stats()  # type: ignore[attr-defined]
    namespace_stats = stats.get("namespaces", {}).get(namespace, {})
    return int(namespace_stats.get("vector_count", 0))


async def _call_api(method: str, path: str, payload: dict) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("testclient", 0),
        "server": ("testserver", 80),
    }

    messages = [
        {"type": "http.request", "body": body, "more_body": False},
    ]
    send_events = []

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):  # type: ignore[no-untyped-def]
        send_events.append(message)

    await app(scope, receive, send)

    status = 500
    response_body = b""
    for event in send_events:
        if event["type"] == "http.response.start":
            status = event["status"]
        elif event["type"] == "http.response.body":
            response_body += event.get("body", b"")

    data = json.loads(response_body.decode("utf-8")) if response_body else {}
    return status, data


def _wait_for_index_new(client: PineconeClient, index_name: str, timeout: float = 180.0, interval: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            index = client.Index(index_name)
            index.describe_index_stats()
            return index
        except NotFoundException:
            time.sleep(interval)
        except Exception:  # pragma: no cover - transient errors
            time.sleep(interval)
    raise RuntimeError(f"Timed out waiting for Pinecone index '{index_name}' to become ready")


def _wait_for_index_legacy(index_name: str, timeout: float = 180.0, interval: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            index = pinecone.Index(index_name)
            index.describe_index_stats()
            return index
        except Exception:  # pragma: no cover - transient or not-ready states
            time.sleep(interval)
    raise RuntimeError(f"Timed out waiting for Pinecone index '{index_name}' to become ready")


def _expected_dimension(settings) -> int:
    if settings.gemini_api_key:
        return settings.pinecone_dimension
    return len(DeterministicEmbedding().embed("probe-dimension"))


@pytest.fixture(scope="module")
def pinecone_index():
    settings = get_settings()
    required_values = [settings.pinecone_api_key, settings.pinecone_index]
    if not all(required_values):
        pytest.skip("Pinecone credentials are not configured", allow_module_level=True)  # type: ignore[arg-type]

    index_name = settings.pinecone_index
    dimension = _expected_dimension(settings)
    metric = settings.pinecone_metric
    cloud = settings.pinecone_cloud
    region = settings.pinecone_region
    pod_type = settings.pinecone_pod_type or "p1.x1"

    if PineconeClient is not None:
        client = PineconeClient(api_key=settings.pinecone_api_key)
        index_names = {item.name for item in client.list_indexes().indexes}
        if index_name in index_names:
            description = client.describe_index(index_name)
            actual_dimension = getattr(description, "dimension", None)
            if actual_dimension is not None and actual_dimension != dimension:
                pytest.skip(
                    f"Pinecone index '{index_name}' dimension {actual_dimension} does not match expected {dimension}. "
                    "Drop or update the index to continue.",
                    allow_module_level=True,
                )  # type: ignore[arg-type]
        else:
            if ServerlessSpec is None:
                pytest.skip(
                    "Pinecone SDK does not expose ServerlessSpec; create the index manually",
                    allow_module_level=True,
                )  # type: ignore[arg-type]
            if not cloud or not region:
                pytest.skip(
                    "Set pinecone_cloud and pinecone_region to auto-create the index",
                    allow_module_level=True,
                )  # type: ignore[arg-type]
            client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        return _wait_for_index_new(client, index_name)

    if not getattr(settings, "pinecone_environment", None):  # pragma: no cover - legacy SDK requires env
        pytest.skip(
            "Pinecone environment is required for legacy pinecone client",
            allow_module_level=True,
        )  # type: ignore[arg-type]

    pinecone.init(api_key=settings.pinecone_api_key, environment=settings.pinecone_environment)  # type: ignore[attr-defined]
    existing = set(pinecone.list_indexes())  # type: ignore[attr-defined]
    if index_name in existing:
        description = pinecone.describe_index(index_name)  # type: ignore[attr-defined]
        actual_dimension = getattr(description, "dimension", None)
        if isinstance(actual_dimension, int) and actual_dimension != dimension:
            pytest.skip(
                f"Pinecone index '{index_name}' dimension {actual_dimension} does not match expected {dimension}. "
                "Drop or update the index to continue.",
                allow_module_level=True,
            )  # type: ignore[arg-type]
    else:
        pinecone.create_index(  # type: ignore[attr-defined]
            name=index_name,
            dimension=dimension,
            metric=metric,
            pod_type=pod_type,
        )
    return _wait_for_index_legacy(index_name)


def _safe_delete(index, namespace: str) -> None:
    try:
        index.delete(namespace=namespace, delete_all=True)  # type: ignore[attr-defined]
    except NotFoundException:
        pass


def test_ingest_file_via_api_real_pinecone(pinecone_index):
    sample_path = Path("tests/data/sample_document.txt").resolve()
    assert sample_path.exists(), "sample document missing"

    org_id = "integration_org"
    branch_id = f"integration_branch_{uuid.uuid4().hex}"  # unique namespace per test run
    namespace = f"{org_id}::{branch_id}"

    _safe_delete(pinecone_index, namespace)
    before_count = _namespace_vector_count(pinecone_index, namespace)

    payload = {
        "context": {
            "org_id": org_id,
            "branch_id": branch_id,
            "user_session_id": f"session-{uuid.uuid4().hex}",
        },
        "documents": [{"source_path": str(sample_path)}],
    }

    status, data = asyncio.run(_call_api("POST", "/api/v1/ingest", payload))

    assert status == 200
    assert data["failed"] == 0
    assert data["processed"] >= 1

    after_count = _namespace_vector_count(pinecone_index, namespace)
    assert after_count > before_count, "expected new vectors ingested into Pinecone"

    _safe_delete(pinecone_index, namespace)
