import json
from unittest.mock import AsyncMock

import groq
import httpx
import pytest

from app.services.groq_service import GroqService, ServiceError


def test_review_history_dashboard_delete(client, mock_review):
    response = client.post(
        "/api/review/code",
        json={"code": "def divide(a,b):\n    return a/b", "language": "python"},
    )
    assert response.status_code == 200, response.text
    row = response.json()
    assert row["findings"][0]["source"] == "groq"
    assert "RAG skipped" in " ".join(row["warnings"])
    assert client.get("/api/reviews").json()[0]["id"] == row["id"]
    assert client.get(
        "/api/reviews", params={"search": "divisor", "language": "python"}
    ).json()
    assert client.get("/api/reviews", params={"search": "nonexistent"}).json() == []
    assert client.get("/api/reviews", params={"language": "java"}).json() == []
    assert client.get(f"/api/reviews/{row['id']}").json()["summary"] == row["summary"]
    stats = client.get("/api/dashboard/stats").json()
    assert stats["total_reviews"] == stats["bugs"] == 1
    assert sum(p["reviews"] for p in client.get("/api/dashboard/activity").json()) == 1
    assert client.delete(f"/api/reviews/{row['id']}").status_code == 204
    assert client.get(f"/api/reviews/{row['id']}").status_code == 404
    assert client.delete(f"/api/reviews/{row['id']}").status_code == 404
    assert client.get("/api/dashboard/stats").json()["total_reviews"] == 0


@pytest.mark.parametrize("code", ["", "  ", "x" * 60001, "a\x00b"])
def test_invalid_code(client, code):
    assert client.post("/api/review/code", json={"code": code}).status_code == 422


def test_no_key_is_honest(client):
    response = client.post(
        "/api/review/code", json={"code": "x = 1", "language": "other"}
    )
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]
    assert client.get("/api/reviews").json() == []


def test_health_empty_status_and_limits(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/api/dashboard/stats").json()["total_reviews"] == 0
    status = client.get("/api/settings/status").json()
    assert not status["groq_connected"]
    assert "api_key" not in json.dumps(status).lower()
    assert client.get("/api/reviews?limit=201").status_code == 422
    assert client.post("/api/review/code", content="x" * 1500001).status_code == 413


@pytest.mark.asyncio
async def test_structured_output_repair(monkeypatch, ai_result):
    complete = AsyncMock(side_effect=['{"summary": 7}', ai_result.model_dump_json()])
    monkeypatch.setattr(GroqService, "_complete", complete)
    assert (await GroqService().review({"files": []})).summary == ai_result.summary
    assert complete.await_count == 2


@pytest.mark.asyncio
async def test_invalid_output_fails(monkeypatch):
    monkeypatch.setattr(GroqService, "_complete", AsyncMock(return_value="not json"))
    with pytest.raises(ServiceError, match="invalid review"):
        await GroqService().review({})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type,status",
    [
        (groq.AuthenticationError, 503),
        (groq.RateLimitError, 429),
        (groq.BadRequestError, 502),
    ],
)
async def test_groq_http_errors(monkeypatch, error_type, status):
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://api.groq.com")
    )
    error = error_type("private provider detail", response=response, body=None)
    complete = AsyncMock(side_effect=error)
    fake = AsyncMock()
    fake.chat.completions.create = complete
    fake.__aenter__.return_value = fake
    monkeypatch.setattr(GroqService, "_client", lambda self: fake)
    with pytest.raises(ServiceError) as caught:
        await GroqService()._complete([])
    assert caught.value.status_code == status
    assert "private provider detail" not in str(caught.value)


@pytest.mark.asyncio
async def test_groq_timeout(monkeypatch):
    fake = AsyncMock()
    fake.__aenter__.return_value = fake
    fake.chat.completions.create.side_effect = groq.APITimeoutError(
        request=httpx.Request("POST", "https://api.groq.com")
    )
    monkeypatch.setattr(GroqService, "_client", lambda self: fake)
    with pytest.raises(ServiceError) as caught:
        await GroqService()._complete([])
    assert caught.value.status_code == 504


@pytest.mark.asyncio
async def test_groq_account_input_limit(monkeypatch):
    fake = AsyncMock()
    fake.__aenter__.return_value = fake
    response = httpx.Response(413, request=httpx.Request("POST", "https://api.groq.com"))
    fake.chat.completions.create.side_effect = groq.APIStatusError(
        "private provider payload", response=response, body=None
    )
    monkeypatch.setattr(GroqService, "_client", lambda self: fake)
    with pytest.raises(ServiceError) as caught:
        await GroqService()._complete([])
    assert caught.value.status_code == 413
    assert "smaller pull request" in str(caught.value)
    assert "private provider payload" not in str(caught.value)
