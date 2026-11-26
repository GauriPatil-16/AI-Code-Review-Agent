import asyncio
import hashlib
import time
from unittest.mock import AsyncMock
from urllib.parse import unquote

import aiosqlite
import httpx
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.agents.fix_agent import build_fix_graph
from app.core.config import get_settings
from app.database.db import SessionLocal, engine
from app.database.models import Review, ReviewSource
from app.schemas.schemas import Finding, FixPlan, FixVerification
from app.services import fix_service, fix_validation
from app.services.github_fix_service import GitHubFixClient, safe_path
from app.services.groq_service import GroqService, ServiceError

ORIGINAL = "def divide(a, b):\n    return a / b\n"
CORRECTED = 'def divide(a, b):\n    if b == 0:\n        raise ValueError("zero divisor")\n    return a / b\n'
SOURCE = {"repository": "acme/repo", "branch": "feature", "sha": "a" * 40}
ISSUE = {
    "title": "Zero divisor",
    "severity": "high",
    "category": "bug",
    "source": "groq",
    "description": "The contract requires rejecting zero divisors with ValueError.",
    "suggestion": "Validate the divisor.",
    "file": "main.py",
    "line": 2,
}


@pytest.fixture
def setup_fix(client, monkeypatch):
    monkeypatch.setattr(
        get_settings(), "github_token", "mock-token-not-a-real-credential"
    )
    plan = FixPlan(
        files=[
            {
                "path": "main.py",
                "code": CORRECTED,
                "issue_ids": [0],
                "explanation": "Reject a zero divisor before division.",
            }
        ],
        skipped_issues=[],
    )
    generate = AsyncMock(return_value=plan)
    verify = AsyncMock(
        return_value=FixVerification(
            issues=[
                {
                    "issue_id": 0,
                    "verdict": "fixed",
                    "evidence": "A zero divisor now raises ValueError before division.",
                }
            ],
            behavior_preserved=True,
            unintended_changes=[],
        )
    )
    monkeypatch.setattr(GroqService, "generate_fixes", generate)
    monkeypatch.setattr(GroqService, "verify_fixes", verify)
    monkeypatch.setattr(fix_validation, "scan_code", AsyncMock(return_value=([], [])))
    monkeypatch.setattr(
        fix_validation,
        "sandbox_tests",
        AsyncMock(
            return_value=[
                {
                    "name": "Existing tests",
                    "status": "skipped",
                    "detail": "No sandbox is available in this mocked test.",
                }
            ]
        ),
    )
    with SessionLocal() as session:
        review = Review(
            title="Divide contract",
            language="python",
            review_type="github",
            source_reference="https://github.com/acme/repo/pull/1",
            original_code=ORIGINAL,
            improved_code="",
            summary="Validate zero.",
            findings=[ISSUE],
            warnings=[],
            context_sources=[],
        )
        session.add(review)
        session.flush()
        session.add(
            ReviewSource(
                review_id=review.id,
                snapshot={
                    "files": [
                        {"path": "main.py", "code": ORIGINAL, "language": "python"}
                    ],
                    "source": SOURCE,
                    "context": [],
                    "documents": [],
                },
            )
        )
        session.commit()
        review_id = review.id
    return {
        "client": client,
        "review_id": review_id,
        "generate": generate,
        "verify": verify,
    }


@pytest.fixture
def github(monkeypatch):
    state = {
        "calls": [],
        "refs": {},
        "prs": [],
        "source_sha": SOURCE["sha"],
        "push": True,
        "lost_pr_response": False,
        "fail_commit_once": False,
    }

    async def request(self, method, path, *, optional=False, **kwargs):
        state["calls"].append((method, path, kwargs))
        if method == "GET" and path == "/user":
            return {"login": "reviewer", "id": 123, "email": None}
        if method == "GET" and path == self.prefix:
            return {"permissions": {"push": state["push"]}}
        if "/git/ref/heads/" in path and method == "GET":
            branch = unquote(path.split("/git/ref/heads/")[1])
            sha = (
                state["source_sha"]
                if branch == "feature"
                else state["refs"].get(branch)
            )
            if sha is None and optional:
                return None
            if sha is None:
                raise ServiceError("Inaccessible branch", 409)
            return {"object": {"sha": sha}}
        if method == "GET" and "/git/trees/" in path:
            raw = ORIGINAL.encode()
            return {
                "tree": [
                    {
                        "path": "main.py",
                        "type": "blob",
                        "mode": "100644",
                        "sha": hashlib.sha1(
                            b"blob " + str(len(raw)).encode() + b"\0" + raw
                        ).hexdigest(),
                    }
                ]
            }
        if method == "GET" and path.endswith("/pulls"):
            return list(state["prs"])
        if method == "GET" and "/git/commits/" in path:
            return {"tree": {"sha": "d" * 40}}
        body = kwargs.get("json", {})
        if method == "POST" and path.endswith("/git/refs"):
            state["refs"][body["ref"].removeprefix("refs/heads/")] = body["sha"]
            return {"ref": body["ref"]}
        if method == "POST" and path.endswith("/git/trees"):
            return {"sha": "b" * 40}
        if method == "POST" and path.endswith("/git/commits"):
            if state["fail_commit_once"]:
                state["fail_commit_once"] = False
                raise ServiceError("Simulated lost commit response", 502)
            return {"sha": "c" * 40}
        if method == "PATCH" and "/git/refs/heads/" in path:
            state["refs"][unquote(path.split("/git/refs/heads/")[1])] = body["sha"]
            return {"object": {"sha": body["sha"]}}
        if method == "POST" and path.endswith("/pulls"):
            pr = {
                "html_url": "https://github.com/acme/repo/pull/2",
                "body": body["body"],
                "head": {
                    "ref": body["head"],
                    "repo": {"full_name": self.source["repository"]},
                },
            }
            state["prs"].append(pr)
            if state["lost_pr_response"]:
                raise ServiceError(
                    "Simulated connection lost after GitHub accepted PR", 502
                )
            return pr
        raise AssertionError(f"Unmocked GitHub operation: {method} {path}")

    monkeypatch.setattr(GitHubFixClient, "request", request)
    return state


def prepare(setup):
    response = setup["client"].post(f"/api/reviews/{setup['review_id']}/fix")
    assert response.status_code == 202, response.text
    workflow_id = response.json()["id"]
    asyncio.run(fix_service.process_one())
    return setup["client"].get(f"/api/fixes/{workflow_id}").json()


def decision(setup, workflow, choice="approve", digest=None):
    return setup["client"].post(
        f"/api/fixes/{workflow['id']}/decision",
        json={"decision": choice, "proposal_hash": digest or workflow["proposal_hash"]},
    )


def writes(github):
    return [c for c in github["calls"] if c[0] != "GET"]


def test_generate_validate_diff_interrupt_and_preserve_source(setup_fix, github):
    workflow = prepare(setup_fix)
    assert workflow["status"] == "Awaiting Approval", workflow
    assert workflow["proposal"]["valid"]
    file = workflow["proposal"]["files"][0]
    assert file["original"] == ORIGINAL and file["corrected"] == CORRECTED
    assert "@@" in file["diff"] and "+    if b == 0:" in file["diff"]
    assert file["added"] == 2
    assert not writes(github)
    with SessionLocal() as session:
        assert (
            session.get(ReviewSource, setup_fix["review_id"]).snapshot["files"][0][
                "code"
            ]
            == ORIGINAL
        )
    assert any(
        v["name"] == "Existing tests" and v["status"] == "skipped"
        for v in workflow["proposal"]["validation"]
    )


def test_approved_workflow_resumes_and_creates_one_branch_commit_pr(setup_fix, github):
    workflow = prepare(setup_fix)
    assert decision(setup_fix, workflow).status_code == 200
    assert not writes(github)
    asyncio.run(fix_service.process_one())
    done = setup_fix["client"].get(f"/api/fixes/{workflow['id']}").json()
    assert done["status"] == "Pull Request Created", done
    assert done["pr_url"] == "https://github.com/acme/repo/pull/2"
    assert done["branch"].startswith("ai-fix/review-")
    assert (
        "feature" not in github["refs"]
        and "main" not in github["refs"]
        and "master" not in github["refs"]
    )
    pr_write = next(c for c in writes(github) if c[1].endswith("/pulls"))
    assert pr_write[2]["json"]["base"] == "feature"
    assert "skipped" in pr_write[2]["json"]["body"]
    assert decision(setup_fix, workflow).status_code == 200
    assert asyncio.run(fix_service.process_one()) is False
    assert sum(c[1].endswith("/git/commits") for c in writes(github)) == 1
    assert sum(c[1].endswith("/pulls") for c in writes(github)) == 1


def test_rejection_discards_and_never_writes(setup_fix, github):
    workflow = prepare(setup_fix)
    assert decision(setup_fix, workflow, "reject").status_code == 200
    asyncio.run(fix_service.process_one())
    row = fix_service.get_workflow(workflow["id"])
    assert (
        row.status == "Rejected"
        and row.finished
        and row.proposal == {"discarded": True}
    )
    assert not writes(github)

    async def checkpoint():
        async with AsyncSqliteSaver.from_conn_string(engine.url.database) as saver:
            return await saver.aget_tuple(
                {"configurable": {"thread_id": "fix-" + workflow["id"]}}
            )

    assert asyncio.run(checkpoint()) is None


def test_no_approval_no_github_writes_even_with_direct_resume(setup_fix, github):
    workflow = prepare(setup_fix)

    async def bypass():
        async with AsyncSqliteSaver.from_conn_string(engine.url.database) as saver:
            graph = build_fix_graph(saver)
            await graph.ainvoke(
                Command(
                    resume={
                        "decision": "approve",
                        "proposal_hash": workflow["proposal_hash"],
                    }
                ),
                {
                    "configurable": {
                        "thread_id": "fix-" + workflow["id"],
                        "lease_owner": "not-an-owner",
                    }
                },
            )

    with pytest.raises(ServiceError):
        asyncio.run(bypass())
    assert not writes(github)


def test_modified_proposal_cannot_be_approved(setup_fix, github):
    workflow = prepare(setup_fix)
    proposal = workflow["proposal"]
    proposal["files"][0]["corrected"] = "print('tampered')\n"
    fix_service.update_workflow(workflow["id"], proposal=proposal)
    assert decision(setup_fix, workflow).status_code == 409
    assert not writes(github)


def test_modification_after_approval_blocks_writes(setup_fix, github):
    workflow = prepare(setup_fix)
    assert decision(setup_fix, workflow).status_code == 200
    proposal = workflow["proposal"]
    proposal["source"]["sha"] = "f" * 40
    fix_service.update_workflow(workflow["id"], proposal=proposal)
    asyncio.run(fix_service.process_one())
    assert fix_service.get_workflow(workflow["id"]).status == "Failed"
    assert not writes(github)


def test_source_changed_requires_fresh_review(setup_fix, github):
    workflow = prepare(setup_fix)
    github["source_sha"] = "f" * 40
    response = decision(setup_fix, workflow)
    assert response.status_code == 409 and "new review" in response.json()["detail"]
    assert fix_service.get_workflow(workflow["id"]).decision is None
    assert not writes(github)


def test_source_moved_after_approval_blocks_publication(setup_fix, github):
    workflow = prepare(setup_fix)
    assert decision(setup_fix, workflow).status_code == 200
    github["source_sha"] = "f" * 40
    asyncio.run(fix_service.process_one())
    assert fix_service.get_workflow(workflow["id"]).status == "Failed"
    assert not writes(github)


def test_permission_failure_and_missing_token(setup_fix, github, monkeypatch):
    workflow = prepare(setup_fix)
    github["push"] = False
    assert decision(setup_fix, workflow).status_code == 403
    monkeypatch.setattr(get_settings(), "github_token", "")
    assert decision(setup_fix, workflow).status_code == 503
    assert not writes(github)


def test_lost_pr_response_is_recovered_without_duplicate(setup_fix, github):
    workflow = prepare(setup_fix)
    github["lost_pr_response"] = True
    assert decision(setup_fix, workflow).status_code == 200
    asyncio.run(fix_service.process_one())
    assert fix_service.get_workflow(workflow["id"]).status == "Pull Request Created"
    assert len(github["prs"]) == 1


def test_invalid_python_is_retried_with_feedback_and_blocked(setup_fix, github):
    setup_fix["generate"].return_value = FixPlan(
        files=[
            {
                "path": "main.py",
                "code": "def broken(:\n",
                "issue_ids": [0],
                "explanation": "Invalid proposal",
            }
        ],
        skipped_issues=[],
    )
    workflow = prepare(setup_fix)
    assert workflow["status"] == "Failed"
    assert setup_fix["generate"].await_count == 2
    assert setup_fix["generate"].call_args.args[0]["validation_feedback"]
    assert any(v["status"] == "failed" for v in workflow["proposal"]["validation"])
    assert setup_fix["verify"].await_count == 0 and not writes(github)


@pytest.mark.parametrize("verdict", ["uncertain", "unsupported", "remaining"])
def test_uncertain_or_unsupported_fixes_are_not_approvable(setup_fix, github, verdict):
    setup_fix["verify"].return_value = FixVerification(
        issues=[
            {
                "issue_id": 0,
                "verdict": verdict,
                "evidence": "Cannot confirm the original issue.",
            }
        ],
        behavior_preserved=True,
        unintended_changes=[],
    )
    workflow = prepare(setup_fix)
    assert workflow["status"] == "Failed" and not workflow["proposal"]["valid"]
    assert not writes(github)


def test_business_logic_regression_blocks_fix(setup_fix, github):
    setup_fix["verify"].return_value = FixVerification(
        issues=[{"issue_id": 0, "verdict": "fixed", "evidence": "Guard exists."}],
        behavior_preserved=False,
        unintended_changes=["Valid input behavior changed."],
    )
    workflow = prepare(setup_fix)
    assert workflow["status"] == "Failed"
    assert any("behavior" in e for e in workflow["proposal"]["errors"])
    assert not writes(github)


def test_new_static_finding_blocks_fix(setup_fix, github, monkeypatch):
    async def scan(code, language, path):
        if code == ORIGINAL:
            return [], []
        return [
            Finding(**{**ISSUE, "title": "F821: Undefined name", "source": "ruff"})
        ], []

    monkeypatch.setattr(fix_validation, "scan_code", scan)
    workflow = prepare(setup_fix)
    assert workflow["status"] == "Failed"
    assert any("new findings" in e for e in workflow["proposal"]["errors"])
    assert not writes(github)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.py",
        "/tmp/outside.py",
        "C:/outside.py",
        "a/../../outside.py",
        "a\\outside.py",
        ".github/workflows/build.yml",
        ".env",
        "a//bad.py",
    ],
)
def test_unsafe_paths_rejected(path):
    with pytest.raises(ServiceError):
        safe_path(path)


def test_path_traversal_from_llm_never_reaches_github(setup_fix, github):
    setup_fix["generate"].return_value = FixPlan(
        files=[
            {
                "path": "../outside.py",
                "code": CORRECTED,
                "issue_ids": [0],
                "explanation": "Bad path",
            }
        ],
        skipped_issues=[],
    )
    workflow = prepare(setup_fix)
    assert workflow["status"] == "Failed" and not writes(github)


def test_duplicate_start_is_one_durable_job(setup_fix):
    a = setup_fix["client"].post(f"/api/reviews/{setup_fix['review_id']}/fix").json()
    b = setup_fix["client"].post(f"/api/reviews/{setup_fix['review_id']}/fix").json()
    assert a["id"] == b["id"]
    claim = fix_service.claim_job()
    assert claim and fix_service.claim_job() is None


def test_cross_origin_approval_is_denied(setup_fix, github):
    workflow = prepare(setup_fix)
    response = setup_fix["client"].post(
        f"/api/fixes/{workflow['id']}/decision",
        headers={"Origin": "https://untrusted.example"},
        json={"decision": "approve", "proposal_hash": workflow["proposal_hash"]},
    )
    assert response.status_code == 403 and not writes(github)


def test_pasted_code_can_be_fixed_but_cannot_create_a_pr(setup_fix, github):
    with SessionLocal() as session:
        row = session.get(ReviewSource, setup_fix["review_id"])
        row.snapshot = {**row.snapshot, "source": None}
        session.commit()
    workflow = prepare(setup_fix)
    assert workflow["status"] == "Awaiting Approval"
    assert decision(setup_fix, workflow).status_code == 422
    assert not writes(github)


def test_multiple_files_and_issues_are_proposed_together(setup_fix):
    with SessionLocal() as session:
        source = session.get(ReviewSource, setup_fix["review_id"])
        source.snapshot = {
            **source.snapshot,
            "files": source.snapshot["files"]
            + [{"path": "other.py", "code": ORIGINAL, "language": "python"}],
        }
        review = session.get(Review, setup_fix["review_id"])
        review.findings = [ISSUE, {**ISSUE, "file": "other.py"}]
        session.commit()
    setup_fix["generate"].return_value = FixPlan(
        files=[
            {
                "path": path,
                "code": CORRECTED,
                "issue_ids": [i],
                "explanation": "Guard zero.",
            }
            for i, path in enumerate(["main.py", "other.py"])
        ],
        skipped_issues=[],
    )
    setup_fix["verify"].return_value = FixVerification(
        issues=[
            {"issue_id": i, "verdict": "fixed", "evidence": "Guard verified."}
            for i in range(2)
        ],
        behavior_preserved=True,
        unintended_changes=[],
    )
    workflow = prepare(setup_fix)
    assert workflow["status"] == "Awaiting Approval"
    assert (
        len(workflow["proposal"]["files"]) == 2
        and len(workflow["proposal"]["addressed_issues"]) == 2
    )


def test_unlocated_finding_is_not_automatically_fixed(setup_fix):
    with SessionLocal() as session:
        row = session.get(Review, setup_fix["review_id"])
        row.findings = [{**ISSUE, "line": None}]
        session.commit()
    assert prepare(setup_fix)["status"] == "Failed"


def test_sandbox_unavailable_is_honest_and_never_executes_code(monkeypatch):
    monkeypatch.setattr(fix_validation.shutil, "which", lambda _: None)
    command = AsyncMock()
    monkeypatch.setattr(fix_validation, "docker_command", command)
    checks = asyncio.run(
        fix_validation.sandbox_tests({"source": SOURCE}, [{"language": "python"}])
    )
    assert checks[0]["status"] == "skipped"
    command.assert_not_awaited()


def test_expired_worker_lease_can_be_recovered(setup_fix):
    row = fix_service.enqueue_fix(setup_fix["review_id"])
    first = fix_service.claim_job()
    assert first and fix_service.claim_job() is None
    fix_service.update_workflow(row.id, lease_until=time.time() - 1)
    second = fix_service.claim_job()
    assert second and first[1] != second[1]


def test_history_and_original_review_remain_available(setup_fix):
    workflow = prepare(setup_fix)
    assert workflow["status"] == "Awaiting Approval"
    detail = setup_fix["client"].get(f"/api/reviews/{setup_fix['review_id']}")
    assert detail.status_code == 200 and detail.json()["original_code"] == ORIGINAL
    assert len(setup_fix["client"].get("/api/reviews").json()) == 1


def test_new_reviews_automatically_queue_fixes(client, mock_review):
    response = client.post(
        "/api/review/code", json={"code": ORIGINAL, "language": "python"}
    )
    assert response.status_code == 200
    workflow = client.get(f"/api/reviews/{response.json()['id']}/fix").json()
    assert workflow["status"] == "Analyzing"


def test_invalid_first_fix_is_regenerated_and_then_validated(setup_fix):
    valid = setup_fix["generate"].return_value
    invalid = FixPlan(
        files=[
            {
                "path": "main.py",
                "code": "def broken(:\n",
                "issue_ids": [0],
                "explanation": "Bad syntax",
            }
        ],
        skipped_issues=[],
    )
    setup_fix["generate"].side_effect = [invalid, valid]
    workflow = prepare(setup_fix)
    assert workflow["status"] == "Awaiting Approval"
    assert workflow["proposal"]["attempts"] == 2


def test_invalid_llm_json_has_a_bounded_schema_repair(monkeypatch):
    completion = AsyncMock(return_value="not JSON")
    monkeypatch.setattr(GroqService, "_complete", completion)
    with pytest.raises(ServiceError, match="schema repair"):
        asyncio.run(GroqService().generate_fixes({"files": []}))
    assert completion.await_count == 2


def test_failed_existing_tests_block_approval(setup_fix, monkeypatch):
    monkeypatch.setattr(
        fix_validation,
        "sandbox_tests",
        AsyncMock(
            return_value=[
                {
                    "name": "Baseline tests",
                    "status": "passed",
                    "detail": "Mocked baseline result",
                },
                {
                    "name": "Corrected tests",
                    "status": "failed",
                    "detail": "Mocked regression result",
                },
            ]
        ),
    )
    workflow = prepare(setup_fix)
    assert workflow["status"] == "Failed" and not workflow["proposal"]["valid"]


def test_removed_python_public_interface_blocks_fix_even_if_ai_approves(setup_fix):
    setup_fix["generate"].return_value = FixPlan(
        files=[
            {
                "path": "main.py",
                "code": "answer = 0\n",
                "issue_ids": [0],
                "explanation": "Remove division",
            }
        ],
        skipped_issues=[],
    )
    workflow = prepare(setup_fix)
    assert workflow["status"] == "Failed"
    assert any("interfaces" in e for e in workflow["proposal"]["errors"])


def test_failed_publication_resumes_same_branch_and_commit_payload(setup_fix, github):
    workflow = prepare(setup_fix)
    assert decision(setup_fix, workflow).status_code == 200
    github["fail_commit_once"] = True
    asyncio.run(fix_service.process_one())
    assert fix_service.get_workflow(workflow["id"]).status == "Failed"
    response = setup_fix["client"].post(f"/api/fixes/{workflow['id']}/retry")
    assert response.status_code == 202
    asyncio.run(fix_service.process_one())
    assert fix_service.get_workflow(workflow["id"]).status == "Pull Request Created"
    assert sum(c[1].endswith("/git/refs") for c in writes(github)) == 1
    commits = [c[2]["json"] for c in writes(github) if c[1].endswith("/git/commits")]
    assert len(commits) == 2 and commits[0] == commits[1]
    assert len(github["prs"]) == 1


def test_approval_can_recover_a_missing_checkpoint_without_regeneration(
    setup_fix, github
):
    workflow = prepare(setup_fix)

    async def remove_checkpoint():
        async with AsyncSqliteSaver.from_conn_string(engine.url.database) as saver:
            await saver.adelete_thread("fix-" + workflow["id"])

    asyncio.run(remove_checkpoint())
    assert decision(setup_fix, workflow).status_code == 200
    asyncio.run(fix_service.process_one())
    assert fix_service.get_workflow(workflow["id"]).status == "Pull Request Created"
    assert setup_fix["generate"].await_count == 1
    assert len(github["prs"]) == 1


def test_review_deletion_revokes_unused_approval(setup_fix, github):
    workflow = prepare(setup_fix)
    assert decision(setup_fix, workflow).status_code == 200
    assert (
        setup_fix["client"].delete(f"/api/reviews/{setup_fix['review_id']}").status_code
        == 204
    )
    asyncio.run(fix_service.process_one())
    assert not writes(github)


def test_sandbox_invocation_is_isolated_and_runs_baseline_and_corrected(monkeypatch):
    monkeypatch.setattr(fix_validation.shutil, "which", lambda _: "docker")
    snapshot = {
        "main.py": ORIGINAL,
        "tests/test_main.py": "def test_example():\n    assert True\n",
    }
    monkeypatch.setattr(
        GitHubFixClient, "validation_snapshot", AsyncMock(return_value=(snapshot, None))
    )
    calls = []

    async def command(*args, **kwargs):
        calls.append(args)
        return 0, "Mocked isolated pytest output: 1 passed"

    monkeypatch.setattr(fix_validation, "docker_command", command)
    checks = asyncio.run(
        fix_validation.sandbox_tests(
            {"source": SOURCE},
            [{"path": "main.py", "language": "python", "corrected": CORRECTED}],
        )
    )
    assert [v["name"] for v in checks] == ["Baseline tests", "Corrected tests"]
    runs = [args for args in calls if args[0] == "run"]
    assert len(runs) == 2
    for args in runs:
        assert (
            "--network=none" in args
            and "--read-only" in args
            and "--cap-drop=ALL" in args
        )
        assert "--user=65534:65534" in args and "--pull=never" in args
        assert "--security-opt=no-new-privileges" in args
        assert not any("GROQ_API_KEY" in arg or "GITHUB_TOKEN" in arg for arg in args)


@pytest.mark.parametrize("status", [401, 403, 404, 409, 422, 500])
def test_github_error_responses_are_handled_without_exposing_provider_bodies(
    monkeypatch, status
):
    async def response(self, method, path, **kwargs):
        return httpx.Response(
            status,
            json={"message": "sensitive mock provider content"},
            request=httpx.Request(method, "https://api.github.com" + path),
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", response)

    async def check():
        async with GitHubFixClient(SOURCE) as github:
            await github.request("GET", github.prefix)

    with pytest.raises(ServiceError) as captured:
        asyncio.run(check())
    assert "sensitive mock provider content" not in str(captured.value)


def test_checkpoint_lock_does_not_block_event_loop_during_status_write(setup_fix):
    from app.agents.fix_agent import generate

    row = fix_service.enqueue_fix(setup_fix["review_id"])

    async def check():
        locked = asyncio.Event()

        async def transient_checkpoint_write():
            async with aiosqlite.connect(engine.url.database) as connection:
                await connection.execute("BEGIN IMMEDIATE")
                await connection.execute(
                    "UPDATE fix_workflows SET status=status WHERE id=?", (row.id,)
                )
                locked.set()
                await asyncio.sleep(0.15)
                await connection.commit()

        task = asyncio.create_task(transient_checkpoint_write())
        await locked.wait()
        start = time.monotonic()
        result = await generate(
            {
                "workflow_id": row.id,
                "snapshot": {"files": [], "context": []},
                "findings": [ISSUE],
                "attempt": 0,
            }
        )
        await task
        assert result["attempt"] == 1
        assert time.monotonic() - start < 5

    asyncio.run(check())
