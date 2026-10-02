from fastapi.testclient import TestClient

from rag_assistant.api import create_app
from rag_assistant.config import Settings
from rag_assistant.pipeline import RAGPipeline


def _client():
    return TestClient(create_app(RAGPipeline(Settings())))


def test_health():
    assert _client().get("/health").json()["status"] == "ok"


def test_ingest_query_delete_flow():
    c = _client()
    r = c.put("/documents/policy", json={"text": "Passwords must be rotated every ninety days."})
    assert r.status_code == 201 and r.json()["chunks"] == 1
    body = c.post("/query", json={"question": "How often are passwords rotated?"}).json()
    assert body["citations"][0]["doc_id"] == "policy"
    assert c.delete("/documents/policy").status_code == 204
    assert c.delete("/documents/policy").status_code == 404


def test_validation_rejects_short_question():
    assert _client().post("/query", json={"question": "hi"}).status_code == 422
