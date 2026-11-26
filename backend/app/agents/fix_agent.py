"""Review continuation: generate, validate, checkpoint, interrupt, resume, publish."""

import asyncio
from typing import TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.core.config import get_settings
from app.database.db import SessionLocal, engine
from app.database.models import Review, ReviewSource
from app.schemas.schemas import FixPlan
from app.services.fix_service import (
    APPROVAL_MESSAGE,
    assert_approved,
    get_workflow,
    proposal_hash,
    update_workflow,
)
from app.services.fix_validation import validate_plan
from app.services.github_fix_service import GitHubFixClient
from app.services.groq_service import GroqService, ServiceError


class FixState(TypedDict, total=False):
    workflow_id: str
    snapshot: dict
    findings: list[dict]
    attempt: int
    plan: dict
    proposal: dict
    approved: bool


def fetch_source(state):
    row = get_workflow(state["workflow_id"])
    with SessionLocal() as session:
        review = session.get(Review, row.review_id)
        source = session.get(ReviewSource, row.review_id)
        if not review or not source:
            raise ServiceError("Review source is unavailable. Run a new review.", 409)
        return {"snapshot": source.snapshot, "findings": review.findings, "attempt": 0}


async def generate(state):
    await asyncio.to_thread(
        update_workflow, state["workflow_id"], status="Generating Fixes"
    )
    plan = await GroqService().generate_fixes(
        {
            "files": state["snapshot"]["files"],
            "context": state["snapshot"].get("context", []),
            "documents": state["snapshot"].get("documents", []),
            "instructions": state["snapshot"].get("instructions", ""),
            "issues": [{"issue_id": i, **f} for i, f in enumerate(state["findings"])],
            "validation_feedback": state.get("proposal", {}).get("errors", []),
            "previous_attempt": state.get("plan"),
            "failed_checks": [
                check
                for check in state.get("proposal", {}).get("validation", [])
                if check["status"] == "failed"
            ],
        }
    )
    return {"plan": plan.model_dump(), "attempt": state["attempt"] + 1}


async def validate(state):
    await asyncio.to_thread(update_workflow, state["workflow_id"], status="Validating")
    try:
        proposal = await validate_plan(
            state["snapshot"], state["findings"], FixPlan.model_validate(state["plan"])
        )
    except ServiceError as error:
        proposal = {
            "valid": False,
            "files": [],
            "validation": [],
            "errors": [str(error)],
            "addressed_issues": [],
            "remaining_issues": state["findings"],
            "source": state["snapshot"].get("source"),
        }
    proposal["attempts"] = state["attempt"]
    await asyncio.to_thread(update_workflow, state["workflow_id"], proposal=proposal)
    return {"proposal": proposal}


def next_stage(state):
    if state["proposal"]["valid"]:
        return "generate_diff"
    return (
        "generate_fixes"
        if state["attempt"] < min(3, max(1, get_settings().fix_max_attempts))
        else "failed_validation"
    )


def generate_diff(state):
    # Diffs were computed from source, not supplied by the model. Bind all
    # displayed changes, validation and the immutable GitHub source to approval.
    digest = proposal_hash(state["proposal"])
    existing = get_workflow(state["workflow_id"])
    if existing.decision:
        if existing.proposal_hash != digest:
            raise ServiceError(
                "Changes differ from the recorded approval. Run a new review.", 409
            )
        return {}
    update_workflow(
        state["workflow_id"],
        proposal=state["proposal"],
        proposal_hash=digest,
        status="Awaiting Approval",
        error=None,
    )
    return {}


def approval(state):
    row = get_workflow(state["workflow_id"])
    response = interrupt(
        {
            "message": APPROVAL_MESSAGE,
            "workflow_id": row.id,
            "proposal_hash": row.proposal_hash,
        }
    )
    row = get_workflow(row.id)
    if (
        response.get("proposal_hash") != row.proposal_hash
        or response.get("decision") != row.decision
    ):
        raise ServiceError(
            "Approval does not match the persisted proposal decision.", 409
        )
    if row.decision == "reject":
        return {"approved": False}
    assert_approved(row.id)
    return {"approved": True}


async def publish_stage(state, config, stage):
    owner = config["configurable"]["lease_owner"]
    current = await asyncio.to_thread(get_workflow, state["workflow_id"])
    if current.status == "Pull Request Created":
        return {}
    row = await asyncio.to_thread(assert_approved, state["workflow_id"], owner)
    await asyncio.to_thread(update_workflow, row.id, status="Creating Pull Request")
    async with GitHubFixClient(row.proposal["source"]) as github:
        github.lease_owner = owner
        url = await github.publish(row.id, stage=stage)
    if url:
        await asyncio.to_thread(
            update_workflow,
            row.id,
            status="Pull Request Created",
            pr_url=url,
            finished=True,
            error=None,
        )
    return {}


async def create_branch(state, config):
    return await publish_stage(state, config, "branch")


async def commit_changes(state, config):
    return await publish_stage(state, config, "commit")


async def create_pull_request(state, config):
    return await publish_stage(state, config, "pr")


def return_result(state):
    row = get_workflow(state["workflow_id"])
    if row.status != "Pull Request Created" or not row.pr_url:
        raise ServiceError("GitHub did not return a completed pull request.", 502)
    return {}


def reject(state):
    update_workflow(state["workflow_id"], status="Rejected", finished=True)
    return {}


def failed(state):
    update_workflow(
        state["workflow_id"],
        status="Failed",
        error="Validation failed after the limited repair attempts. Review the results; GitHub was not changed.",
        finished=True,
    )
    return {}


def build_fix_graph(checkpointer):
    graph = StateGraph(FixState)
    for name, node in [
        ("fetch_source", fetch_source),
        ("generate_fixes", generate),
        ("validate_fixes", validate),
        ("generate_diff", generate_diff),
        ("human_approval", approval),
        ("create_branch", create_branch),
        ("commit_changes", commit_changes),
        ("create_pull_request", create_pull_request),
        ("return_result", return_result),
        ("reject_changes", reject),
        ("failed_validation", failed),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "fetch_source")
    graph.add_edge("fetch_source", "generate_fixes")
    graph.add_edge("generate_fixes", "validate_fixes")
    graph.add_conditional_edges(
        "validate_fixes",
        next_stage,
        ["generate_diff", "generate_fixes", "failed_validation"],
    )
    graph.add_edge("generate_diff", "human_approval")
    graph.add_conditional_edges(
        "human_approval",
        lambda state: "create_branch" if state["approved"] else "reject_changes",
    )
    graph.add_edge("create_branch", "commit_changes")
    graph.add_edge("commit_changes", "create_pull_request")
    graph.add_edge("create_pull_request", "return_result")
    for name in ("return_result", "reject_changes", "failed_validation"):
        graph.add_edge(name, END)
    return graph.compile(checkpointer=checkpointer)


async def execute_workflow(workflow_id, owner):
    config = {
        "configurable": {"thread_id": "fix-" + workflow_id, "lease_owner": owner},
        "recursion_limit": 25,
    }
    async with AsyncSqliteSaver.from_conn_string(engine.url.database) as saver:
        await saver.conn.execute("PRAGMA busy_timeout=30000")
        graph = build_fix_graph(saver)
        checkpoint = await graph.aget_state(config)
        row = await asyncio.to_thread(get_workflow, workflow_id)
        if row.decision:
            if row.status == "Pull Request Created":
                return
            if checkpoint.next not in (
                ("human_approval",),
                ("create_branch",),
                ("commit_changes",),
                ("create_pull_request",),
                ("return_result",),
                ("reject_changes",),
            ):
                # The proposal row can be committed immediately before the
                # checkpoint. Recover that narrow crash window using the exact
                # durable proposal, without regenerating any approved source.
                await graph.aupdate_state(
                    config,
                    {"workflow_id": workflow_id, "proposal": row.proposal},
                    as_node="generate_diff",
                )
                await graph.ainvoke(None, config)
                checkpoint = await graph.aget_state(config)
            elif "human_approval" in checkpoint.next and not checkpoint.interrupts:
                await graph.ainvoke(None, config)
                checkpoint = await graph.aget_state(config)
            if checkpoint.next:
                value = (
                    Command(
                        resume={
                            "decision": row.decision,
                            "proposal_hash": row.proposal_hash,
                        }
                    )
                    if "human_approval" in checkpoint.next
                    else None
                )
                await graph.ainvoke(value, config)
            if row.decision == "reject":
                await saver.adelete_thread("fix-" + workflow_id)
                await asyncio.to_thread(
                    update_workflow,
                    workflow_id,
                    proposal={"discarded": True},
                    finished=True,
                )
        elif checkpoint.next:
            await graph.ainvoke(None, config)
        elif not checkpoint.values:
            await graph.ainvoke({"workflow_id": workflow_id}, config)
