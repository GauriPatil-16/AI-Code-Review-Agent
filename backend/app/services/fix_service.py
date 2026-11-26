"""Durable fix jobs, exact-content approval, and an atomic SQLite worker lease."""

import asyncio
import hashlib
import json
import logging
import time
from contextlib import suppress
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.database.db import SessionLocal
from app.database.models import FixWorkflow, Review, ReviewSource
from app.schemas.schemas import FixWorkflowResponse
from app.services.github_fix_service import GitHubFixClient
from app.services.groq_service import ServiceError

APPROVAL_MESSAGE = "AI has generated code fixes. Please review and approve the changes to create a GitHub Pull Request."


def proposal_hash(proposal):
    return hashlib.sha256(
        json.dumps(
            proposal, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def get_workflow(workflow_id):
    with SessionLocal() as session:
        row = session.get(FixWorkflow, workflow_id)
        if row is None:
            raise ServiceError("Fix workflow not found.", 404)
        return row


def serialize_workflow(row):
    return FixWorkflowResponse.model_validate(
        {field: getattr(row, field) for field in FixWorkflowResponse.model_fields}
    )


def update_workflow(workflow_id, **changes):
    with SessionLocal() as session:
        row = session.get(FixWorkflow, workflow_id)
        if row is None:
            raise ServiceError("Fix workflow not found.", 404)
        for name, value in changes.items():
            setattr(row, name, value)
        session.commit()
        return row


def enqueue_fix(review_id):
    with SessionLocal() as session:
        review = session.get(Review, review_id)
        if not review:
            raise ServiceError("Review not found.", 404)
        existing = session.scalar(
            select(FixWorkflow).where(FixWorkflow.review_id == review_id)
        )
        if existing:
            return existing
        if not session.get(ReviewSource, review_id):
            raise ServiceError(
                "This review predates source snapshots. Run a new review before generating fixes.",
                409,
            )
        if not review.findings:
            raise ServiceError("There are no identified issues to fix.", 422)
        row = FixWorkflow(review_id=review_id, status="Analyzing")
        session.add(row)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            return session.scalar(
                select(FixWorkflow).where(FixWorkflow.review_id == review_id)
            )
        return row


def assert_approved(workflow_id, lease_owner=None):
    row = get_workflow(workflow_id)
    with SessionLocal() as session:
        if session.get(Review, row.review_id) is None:
            raise ServiceError(
                "The review was deleted; its GitHub approval is no longer usable.", 409
            )
    if (
        row.decision != "approve"
        or row.status not in ("Approved", "Creating Pull Request")
        or not row.approved_hash
        or row.approved_hash != row.proposal_hash
        or proposal_hash(row.proposal) != row.approved_hash
        or not row.proposal.get("valid")
        or not row.proposal.get("files")
        or not row.proposal.get("source")
    ):
        raise ServiceError(
            "GitHub writes require approval of these exact validated files and source commit.",
            409,
        )
    if lease_owner is not None and (
        row.lease_owner != lease_owner or row.lease_until <= time.time()
    ):
        raise ServiceError(
            "Workflow ownership changed; this worker cannot write to GitHub.", 409
        )
    return row


async def decide(workflow_id, decision):
    row = await asyncio.to_thread(get_workflow, workflow_id)
    if (
        row.decision == "reject"
        and decision.decision == "reject"
        and row.proposal_hash == decision.proposal_hash
    ):
        return row
    if (
        decision.proposal_hash != row.proposal_hash
        or proposal_hash(row.proposal) != row.proposal_hash
    ):
        raise ServiceError(
            "The proposal changed. Refresh and review the current changes before approving.",
            409,
        )
    if row.decision:
        if row.decision != decision.decision:
            raise ServiceError(
                "This workflow already has a different approval decision.", 409
            )
        return row
    if row.status != "Awaiting Approval" or not row.proposal.get("valid"):
        raise ServiceError(
            "Only a validated workflow awaiting approval can receive a decision.", 409
        )
    if decision.decision == "approve":
        if not row.proposal.get("source"):
            raise ServiceError(
                "Pasted code has no GitHub source branch. Copy the validated fix instead.",
                422,
            )
        async with GitHubFixClient(row.proposal["source"]) as github:
            await github.preflight(row.proposal["files"])

    def record_decision():
        # Recheck the hash/status inside the atomic update after the network preflight.
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with SessionLocal() as session:
            count = session.execute(
                update(FixWorkflow)
                .where(
                    FixWorkflow.id == workflow_id,
                    FixWorkflow.status == "Awaiting Approval",
                    FixWorkflow.decision.is_(None),
                    FixWorkflow.proposal_hash == decision.proposal_hash,
                    FixWorkflow.proposal == row.proposal,
                )
                .values(
                    decision=decision.decision,
                    approved_hash=decision.proposal_hash
                    if decision.decision == "approve"
                    else None,
                    approved_at=now,
                    status="Approved" if decision.decision == "approve" else "Rejected",
                    proposal=row.proposal
                    if decision.decision == "approve"
                    else {"discarded": True},
                    finished=False,
                )
            ).rowcount
            session.commit()
        current = get_workflow(workflow_id)
        if (
            current.decision != "reject"
            and proposal_hash(current.proposal) != decision.proposal_hash
        ):
            raise ServiceError(
                "The proposal changed while approval was being recorded. Review it again.",
                409,
            )
        if not count and current.decision != decision.decision:
            raise ServiceError("Another decision has already been recorded.", 409)
        return current

    return await asyncio.to_thread(record_decision)


def claim_job():
    now, owner = time.time(), uuid4().hex
    with SessionLocal() as session:
        rows = session.scalars(
            select(FixWorkflow)
            .where(
                FixWorkflow.finished.is_(False),
                FixWorkflow.lease_until < now,
                FixWorkflow.status.in_(
                    [
                        "Analyzing",
                        "Generating Fixes",
                        "Validating",
                        "Approved",
                        "Creating Pull Request",
                        "Rejected",
                    ]
                ),
            )
            .order_by(FixWorkflow.created_at)
            .limit(10)
        ).all()
        for row in rows:
            count = session.execute(
                update(FixWorkflow)
                .where(
                    FixWorkflow.id == row.id,
                    FixWorkflow.lease_until < now,
                    FixWorkflow.finished.is_(False),
                )
                .values(lease_owner=owner, lease_until=now + 120)
            ).rowcount
            session.commit()
            if count:
                return row.id, owner
    return None


async def heartbeat(workflow_id, owner):
    def renew():
        with SessionLocal() as session:
            session.execute(
                update(FixWorkflow)
                .where(FixWorkflow.id == workflow_id, FixWorkflow.lease_owner == owner)
                .values(lease_until=time.time() + 120)
            )
            session.commit()

    while True:
        await asyncio.sleep(15)
        await asyncio.to_thread(renew)


async def process_one():
    claim = await asyncio.to_thread(claim_job)
    if not claim:
        return False
    workflow_id, owner = claim
    beat = asyncio.create_task(heartbeat(workflow_id, owner))

    def mark_failed(message):
        # A delayed worker must not overwrite a newer worker's result.
        with SessionLocal() as session:
            session.execute(
                update(FixWorkflow)
                .where(FixWorkflow.id == workflow_id, FixWorkflow.lease_owner == owner)
                .values(status="Failed", error=message, finished=True)
            )
            session.commit()

    try:
        from app.agents.fix_agent import execute_workflow

        await execute_workflow(workflow_id, owner)
    except ServiceError as error:
        await asyncio.to_thread(mark_failed, str(error))
    except Exception as error:
        # Provider errors and submitted code are never printed to logs.
        logging.getLogger(__name__).warning(
            "Fix workflow failed (%s)", type(error).__name__
        )
        await asyncio.to_thread(
            mark_failed,
            "The fix workflow failed unexpectedly. No unapproved GitHub writes are permitted.",
        )
    finally:
        beat.cancel()
        with suppress(asyncio.CancelledError):
            await beat

        def release():
            with SessionLocal() as session:
                session.execute(
                    update(FixWorkflow)
                    .where(
                        FixWorkflow.id == workflow_id, FixWorkflow.lease_owner == owner
                    )
                    .values(lease_until=0, lease_owner=None)
                )
                session.commit()
                return session.get(FixWorkflow, workflow_id) is not None

        if not await asyncio.to_thread(release):
            await delete_checkpoints(workflow_id)
    return True


async def worker():
    while True:
        try:
            if not await process_one():
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(2)


async def retry_publish(workflow_id):
    row = await asyncio.to_thread(get_workflow, workflow_id)
    if (
        row.status != "Failed"
        or row.decision != "approve"
        or row.approved_hash != proposal_hash(row.proposal)
    ):
        raise ServiceError(
            "Only an unchanged, previously approved publication can be retried. Otherwise run a new review.",
            409,
        )
    async with GitHubFixClient(row.proposal["source"]) as github:
        await github.preflight(row.proposal["files"])

    def record_retry():
        with SessionLocal() as session:
            session.execute(
                update(FixWorkflow)
                .where(FixWorkflow.id == workflow_id, FixWorkflow.status == "Failed")
                .values(status="Approved", finished=False, error=None)
            )
            session.commit()
        return get_workflow(workflow_id)

    return await asyncio.to_thread(record_retry)


async def delete_checkpoints(workflow_id):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from app.database.db import engine

    async with AsyncSqliteSaver.from_conn_string(engine.url.database) as saver:
        await saver.conn.execute("PRAGMA busy_timeout=30000")
        await saver.adelete_thread("fix-" + workflow_id)
