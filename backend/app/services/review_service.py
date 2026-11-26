import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.db import SessionLocal
from app.database.models import Review, ReviewSource
from app.schemas.schemas import CodeRequest, PRRequest, ReviewResponse
from app.services.github_service import fetch_pull_request
from app.services.groq_service import ServiceError


def serialize(review: Review) -> ReviewResponse:
    return ReviewResponse.model_validate(
        {field: getattr(review, field) for field in ReviewResponse.model_fields}
    )


def save_review(state: dict) -> dict:
    ai = state["ai"]
    with SessionLocal() as session:
        row = Review(
            title=state["title"],
            language=state["language"],
            review_type=state["review_type"],
            source_reference=state.get("source_reference"),
            original_code=state["code"],
            summary=ai.summary,
            improved_code=ai.improved_code,
            findings=[f.model_dump() for f in state["findings"]],
            warnings=list(dict.fromkeys(state["warnings"])),
            context_sources=state.get("context_sources", []),
        )
        session.add(row)
        session.flush()
        session.add(ReviewSource(review_id=row.id, snapshot={
            "files": state["files"],
            "documents": state.get("documents", []),
            "context": state.get("context", []),
            "instructions": state.get("instructions", ""),
            "repository": state.get("repository"),
            "source": state.get("source"),
        }))
        session.commit()
        session.refresh(row)
        return serialize(row).model_dump(mode="json")


async def review_code(request: CodeRequest) -> dict:
    from app.agents.review_agent import run_review

    state = {
        "code": request.code,
        "language": request.language,
        "instructions": request.instructions,
        "title": f"{request.language.title()} code review",
        "review_type": "code",
        "source_reference": None,
        "repository_id": request.repository_id,
        "documents": [d.model_dump() for d in request.documents],
        "files": [
            {"path": "snippet", "code": request.code, "language": request.language}
        ],
        "warnings": [],
    }
    try:
        return await asyncio.wait_for(run_review(state), get_settings().review_timeout)
    except TimeoutError:
        raise ServiceError(
            "Review workflow timed out. Reduce the input and retry.", 504
        ) from None


async def review_pull_request(request: PRRequest) -> dict:
    from app.agents.review_agent import run_review

    async def run():
        pr = await fetch_pull_request(request.url)
        return await run_review(
            {
                **pr,
                "code": "\n\n".join(
                    f"// FILE: {f['path']}\n{f['code']}" for f in pr["files"]
                ),
                "instructions": request.instructions,
                "review_type": "github",
            }
        )

    try:
        return await asyncio.wait_for(run(), get_settings().review_timeout)
    except TimeoutError:
        raise ServiceError(
            "PR review timed out. Try a smaller pull request.", 504
        ) from None


def list_reviews(
    session: Session,
    search: str = "",
    language: str = "",
    limit: int = 100,
    offset: int = 0,
):
    query = select(Review)
    if search:
        query = query.where(
            Review.title.contains(search, autoescape=True)
            | Review.summary.contains(search, autoescape=True)
        )
    if language:
        query = query.where(Review.language == language)
    return [
        serialize(r)
        for r in session.scalars(
            query.order_by(Review.created_at.desc()).offset(offset).limit(limit)
        )
    ]


def statistics(session: Session) -> dict:
    rows = session.scalars(select(Review)).all()
    findings = [f for row in rows for f in row.findings]
    return {
        "total_reviews": len(rows),
        "bugs": sum(f["category"] == "bug" for f in findings),
        "security": sum(f["category"] == "security" for f in findings),
        "improvements": sum(
            f["category"] in ("quality", "performance") for f in findings
        ),
        "severity": {
            s: sum(f["severity"] == s for f in findings)
            for s in ["critical", "high", "medium", "low"]
        },
    }


def activity(session: Session) -> list[dict]:
    today = datetime.now(timezone.utc).date()
    dates = [today - timedelta(days=i) for i in range(6, -1, -1)]
    rows = session.scalars(
        select(Review).where(
            Review.created_at >= datetime.combine(dates[0], datetime.min.time())
        )
    ).all()
    return [
        {
            "date": d.isoformat(),
            "label": d.strftime("%a"),
            "reviews": sum(r.created_at.date() == d for r in rows),
        }
        for d in dates
    ]
