import io
import time


def _wait_for_job(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/documents/{job_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("done", "failed"):
            return data
        time.sleep(0.3)
    raise TimeoutError("job did not finish in time")


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_upload_rejects_unsupported_type(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("notes.md", io.BytesIO(b"# hi"), "text/markdown")},
    )
    assert resp.status_code == 400


def test_upload_rejects_empty_file(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert resp.status_code == 400


def test_query_validation_rejects_short_question(client):
    resp = client.post("/query", json={"question": "ab"})
    assert resp.status_code == 422


def test_upload_ingest_and_query_roundtrip(client):
    content = (
        b"The vault opens at 9am and closes at 5pm on weekdays.\n\n"
        b"Only employees with badge level 3 or higher may enter the vault.\n\n"
        b"The vault was last audited in March."
    )
    resp = client.post(
        "/documents/upload",
        files={"file": ("vault_policy.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"

    job = _wait_for_job(client, body["job_id"])
    assert job["status"] == "done"
    assert job["chunk_count"] >= 1

    docs = client.get("/documents").json()
    assert any(d["document_id"] == body["document_id"] for d in docs)

    q = client.post("/query", json={"question": "What badge level is needed to enter the vault?"})
    assert q.status_code == 200
    data = q.json()
    assert "sources" in data and len(data["sources"]) > 0
    assert data["metrics"]["generation_mode"] in ("openai", "extractive_fallback", "not_found")
    assert data["metrics"]["retrieval_latency_ms"] >= 0


def test_query_with_no_documents_returns_404(client, monkeypatch):
    # scoping to a document_id that can't exist forces an empty retrieval result
    resp = client.post("/query", json={"question": "irrelevant question here", "document_ids": ["nope"]})
    assert resp.status_code == 404
