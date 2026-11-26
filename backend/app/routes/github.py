import hashlib
import hmac
import json

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import ValidationError

from app.core.config import get_settings
from app.schemas.schemas import PRRequest
from app.services.groq_service import ServiceError
from app.services.review_service import review_pull_request

router = APIRouter(tags=["github"])


async def webhook_review(url: str):
    try:
        await review_pull_request(PRRequest(url=url))
    except (ServiceError, ValidationError):
        # Do not log source, URLs with credentials or provider payloads.
        import logging

        logging.getLogger(__name__).warning(
            "Webhook review failed; retry manually in the review UI."
        )


@router.post("/api/webhooks/github", status_code=202)
async def webhook(request: Request, background: BackgroundTasks):
    secret = get_settings().github_webhook_secret
    if not secret:
        raise HTTPException(503, "GitHub webhook secret is not configured.")
    body = await request.body()
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(
        expected, request.headers.get("X-Hub-Signature-256", "")
    ):
        raise HTTPException(401, "Invalid webhook signature.")
    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(400, "Invalid webhook JSON.") from None
    if not isinstance(payload, dict):
        raise HTTPException(422, "Expected a JSON object.")
    event = request.headers.get("X-GitHub-Event")
    if event == "ping":
        return {"status": "pong"}
    if event != "pull_request" or payload.get("action") not in (
        "opened",
        "synchronize",
        "reopened",
    ):
        return {"status": "ignored"}
    try:
        pr = PRRequest(url=payload["pull_request"]["html_url"])
    except (KeyError, TypeError, ValidationError):
        raise HTTPException(422, "Invalid pull request payload.") from None
    background.add_task(webhook_review, pr.url)
    return {
        "status": "accepted",
        "message": "Review scheduled in this process. No GitHub comment will be posted.",
    }
