import base64
import hashlib
import hmac
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from app.core.config import get_settings
from app.services.github_service import fetch_pull_request
from app.services.groq_service import ServiceError


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/a/b/pull/1",
        "https://evil.com/a/b/pull/1",
        "https://github.com.evil.com/a/b/pull/1",
        "https://user@github.com/a/b/pull/1",
        "https://github.com/a/b/pull/0",
        "https://github.com/a/b/pull/1?x=y",
        "https://github.com/a/b/issues/1",
    ],
)
def test_reject_bad_url(client, url):
    assert client.post("/api/review/github", json={"url": url}).status_code == 422


@pytest.mark.asyncio
async def test_fetch_pr_with_context(monkeypatch):
    async def get(self, url, **kwargs):
        request = httpx.Request("GET", "https://api.github.com" + url)
        if url.endswith("/pulls/1"):
            body = {
                "title": "Fix calculation",
                "head": {"sha": "abc"},
                "changed_files": 3,
            }
        elif url.endswith("/files"):
            body = [
                {
                    "filename": "app.py",
                    "status": "modified",
                    "patch": "@@ -1 +1 @@\n+x = 1",
                    "changes": 1,
                },
                {"filename": "logo.png", "status": "added"},
                {"filename": "old.py", "status": "removed"},
            ]
        elif url.endswith("app.py"):
            body = {
                "encoding": "base64",
                "size": 5,
                "content": base64.b64encode(b"x = 1").decode(),
            }
        else:
            return httpx.Response(404, request=request)
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    result = await fetch_pull_request("https://github.com/a/b/pull/1")
    assert result["files"][0]["code"] == "x = 1"
    assert result["repository_id"] == "a/b@abc"
    assert len(result["warnings"]) == 2
    assert result["reviewed_files"] == 1


@pytest.mark.asyncio
async def test_github_denied(monkeypatch):
    async def denied(self, *args, **kwargs):
        return httpx.Response(
            403, request=httpx.Request("GET", "https://api.github.com")
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", denied)
    with pytest.raises(ServiceError, match="access denied"):
        await fetch_pull_request("https://github.com/a/b/pull/1")


def signed(body):
    return "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()


@pytest.mark.asyncio
async def test_fork_review_reads_source_from_the_head_repository(monkeypatch):
    paths = []
    async def get(self, url, **kwargs):
        paths.append(url)
        request = httpx.Request("GET", "https://api.github.com" + url)
        if url.endswith("/pulls/1"):
            body = {"title": "Fork fix", "head": {"sha": "a" * 40, "ref": "feature", "repo": {"full_name": "contributor/fork"}}, "changed_files": 1}
        elif url.endswith("/files"):
            body = [{"filename": "app.py", "status": "modified", "patch": "+x = 1", "changes": 1}]
        elif url == "/repos/contributor/fork/contents/app.py":
            body = {"encoding": "base64", "size": 6, "content": base64.b64encode(b"x = 1\n").decode(), "sha": "b" * 40}
        else:
            return httpx.Response(404, request=request)
        return httpx.Response(200, json=body, request=request)
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    result = await fetch_pull_request("https://github.com/upstream/repo/pull/1")
    assert result["source"] == {"repository": "contributor/fork", "branch": "feature", "sha": "a" * 40}
    assert "/repos/contributor/fork/contents/app.py" in paths
    assert "/repos/upstream/repo/contents/app.py" not in paths


def test_webhook_signature_and_review(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "github_webhook_secret", "test-secret")
    import app.routes.github as route

    stub = AsyncMock(return_value={})
    monkeypatch.setattr(route, "review_pull_request", stub)
    body = json.dumps(
        {
            "action": "opened",
            "pull_request": {"html_url": "https://github.com/a/b/pull/1"},
        }
    ).encode()
    assert client.post("/api/webhooks/github", content=body).status_code == 401
    response = client.post(
        "/api/webhooks/github",
        content=body,
        headers={"X-Hub-Signature-256": signed(body), "X-GitHub-Event": "pull_request"},
    )
    assert response.status_code == 202
    assert stub.await_count == 1
    body = b"{}"
    assert (
        client.post(
            "/api/webhooks/github",
            content=body,
            headers={"X-Hub-Signature-256": signed(body), "X-GitHub-Event": "ping"},
        ).json()["status"]
        == "pong"
    )


def test_webhook_unconfigured(client):
    assert client.post("/api/webhooks/github", json={}).status_code == 503


def test_pr_endpoint_workflow(client, mock_review, monkeypatch):
    import app.services.review_service as service

    pr = {
        "title": "Fix divide",
        "language": "python",
        "source_reference": "https://github.com/a/b/pull/1",
        "repository_id": None,
        "files": [
            {
                "path": "main.py",
                "code": "def divide(a,b):\n return a/b",
                "language": "python",
                "diff": "+ return a/b",
            }
        ],
        "documents": [],
        "warnings": ["One file reviewed."],
    }
    monkeypatch.setattr(service, "fetch_pull_request", AsyncMock(return_value=pr))
    response = client.post(
        "/api/review/github", json={"url": "https://github.com/a/b/pull/1"}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["review_type"] == "github"
    assert result["source_reference"] == pr["source_reference"]
    assert result["improved_code"] == ""
    assert client.get("/api/reviews").json()[0]["title"] == "Fix divide"
