from fastapi import APIRouter, Depends, Header
from sqlalchemy import select

from app.core.config import get_settings
from app.database.db import SessionLocal
from app.database.models import FixWorkflow
from app.schemas.schemas import FixDecision, FixWorkflowResponse
from app.services.fix_service import (
    decide,
    enqueue_fix,
    get_workflow,
    retry_publish,
    serialize_workflow,
)
from app.services.groq_service import ServiceError


def trusted_origin(origin: str | None = Header(default=None)):
    if origin and origin not in get_settings().cors_origins:
        raise ServiceError(
            "This origin cannot approve changes or start fix workflows.", 403
        )


router = APIRouter(tags=["fixes"], dependencies=[Depends(trusted_origin)])


@router.post(
    "/api/reviews/{review_id}/fix", response_model=FixWorkflowResponse, status_code=202
)
def start(review_id: str):
    return serialize_workflow(enqueue_fix(review_id))


@router.get("/api/reviews/{review_id}/fix", response_model=FixWorkflowResponse | None)
def for_review(review_id: str):
    with SessionLocal() as session:
        row = session.scalar(
            select(FixWorkflow).where(FixWorkflow.review_id == review_id)
        )
        return serialize_workflow(row) if row else None


@router.get("/api/fixes/{workflow_id}", response_model=FixWorkflowResponse)
def detail(workflow_id: str):
    return serialize_workflow(get_workflow(workflow_id))


@router.post("/api/fixes/{workflow_id}/decision", response_model=FixWorkflowResponse)
async def decision(workflow_id: str, request: FixDecision):
    return serialize_workflow(await decide(workflow_id, request))


@router.post(
    "/api/fixes/{workflow_id}/retry",
    response_model=FixWorkflowResponse,
    status_code=202,
)
async def retry(workflow_id: str):
    return serialize_workflow(await retry_publish(workflow_id))
