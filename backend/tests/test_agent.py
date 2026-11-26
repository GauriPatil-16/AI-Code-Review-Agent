from unittest.mock import AsyncMock

import pytest

from app.agents.review_agent import validate_findings
from app.rag import rag_service
from app.services.analysis_service import scan_code

import json
import os
from pathlib import Path

EVALUATIONS = json.loads(
    (Path(__file__).parent / "evaluation_cases.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", EVALUATIONS, ids=lambda c: c["id"])
def test_seeded_case_workflow(client, monkeypatch, case):
    """Offline contract evaluation with an explicit oracle; does not measure LLM recall."""
    from app.schemas.schemas import AIReview
    from app.services.groq_service import GroqService
    from app.agents import review_agent

    result = AIReview(
        summary=case["instructions"],
        improved_code=case["code"],
        findings=[
            {
                "title": case["keywords"][0] + " issue",
                "severity": "high",
                "category": case["category"],
                "description": "Seeded evaluation oracle, not a live AI finding.",
                "suggestion": case["instructions"],
                "line": case["line"],
                "file": "snippet",
                "source": "groq",
            }
        ],
    )
    monkeypatch.setattr(GroqService, "review", AsyncMock(return_value=result))
    monkeypatch.setattr(review_agent, "scan_code", AsyncMock(return_value=([], [])))
    response = client.post(
        "/api/review/code",
        json={k: case[k] for k in ("code", "language", "instructions")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["findings"][0]["category"] == case["category"]
    assert response.json()["findings"][0]["line"] == case["line"]
    assert client.get("/api/dashboard/stats").json()["total_reviews"] == 1


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_EVAL") != "1",
    reason="Set RUN_LIVE_EVAL=1 and LIVE_GROQ_API_KEY to evaluate Groq quality (billable).",
)
@pytest.mark.parametrize("case", EVALUATIONS, ids=lambda c: c["id"])
@pytest.mark.asyncio
async def test_live_detection_quality(monkeypatch, case):
    from app.core.config import get_settings
    from app.services.groq_service import GroqService

    key = os.environ.get("LIVE_GROQ_API_KEY", "")
    assert key, "LIVE_GROQ_API_KEY must be configured for live evaluation"
    monkeypatch.setattr(get_settings(), "groq_api_key", key)
    result = await GroqService().review(
        {
            "language": case["language"],
            "source_kind": "code",
            "files": [{"path": "snippet", "code": case["code"]}],
            "custom_instructions": case["instructions"],
            "retrieved_context": [],
            "scanner_findings": [],
        }
    )
    matched = [
        f
        for f in result.findings
        if f.category == case["category"]
        and any(k in (f.title + " " + f.description).lower() for k in case["keywords"])
    ]
    assert matched, (
        f"Expected seeded finding for {case['id']}; got {result.model_dump()}"
    )


def test_line_validation(ai_result):
    ai_result.findings[0].line = 999
    state = {
        "ai": ai_result,
        "files": [{"path": "snippet", "code": "pass"}],
        "static_findings": [],
        "warnings": [],
        "review_type": "code",
    }
    result = validate_findings(state)
    assert result["findings"][0].line is None
    assert result["warnings"]


def test_no_save_when_generation_fails(client, monkeypatch):
    from app.services.groq_service import GroqService, ServiceError

    monkeypatch.setattr(
        GroqService, "review", AsyncMock(side_effect=ServiceError("Invalid output"))
    )
    response = client.post("/run", json={"code": "x=1", "language": "other"})
    assert response.status_code == 502
    assert client.get("/api/reviews").json() == []


def test_graph_rag_context(client, mock_review):
    response = client.post(
        "/api/review/code",
        json={
            "code": "def divide(a,b):\n return a/b",
            "repository_id": "demo",
            "documents": [
                {
                    "path": "guidelines.md",
                    "content": "divide must validate a divisor of zero before a/b.",
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert any("guidelines.md" in s for s in response.json()["context_sources"])


def test_rag_index_and_isolation(client):
    docs = [
        {
            "path": "rules.md",
            "content": "Validate authorization before deleting records.",
        }
    ]
    response = client.post(
        "/api/rag/index", json={"repository_id": "auth-repo", "documents": docs}
    )
    assert response.status_code == 200, response.text
    found = rag_service.retrieve("auth-repo", "authorization deleting records")
    assert found and found[0]["source"] == "rules.md@0"
    assert (
        rag_service.retrieve("different-repo", "authorization deleting records") == []
    )


def test_rag_update_removes_stale_chunks():
    rag_service.index_documents(
        "update-test", [{"path": "a.md", "content": "authorization " * 300}]
    )
    rag_service.index_documents(
        "update-test", [{"path": "a.md", "content": "authorization"}]
    )
    records = rag_service.collection().get(where={"repository_id": "update-test"})
    assert len(records["ids"]) == 1


def test_rag_guideline_also_present_in_changed_files():
    """A PR may change CONTRIBUTING.md while also fetching it as a guideline."""
    documents = [
        {"path": "CONTRIBUTING.md", "content": "Older coding guidelines."},
        {"path": "CONTRIBUTING.md", "content": "Current coding guidelines."},
    ]
    count = rag_service.index_documents("duplicate-guideline", documents)
    records = rag_service.collection().get(
        where={"repository_id": "duplicate-guideline"}
    )
    assert count == 1
    assert records["documents"] == ["Current coding guidelines."]


@pytest.mark.asyncio
async def test_real_ruff_scan():
    import shutil

    if not shutil.which("ruff"):
        pytest.skip("Ruff unavailable on PATH")
    findings, _ = await scan_code("import os\n", "python")
    assert any(f.source == "ruff" and "F401" in f.title for f in findings)


@pytest.mark.asyncio
async def test_missing_scanners(monkeypatch):
    monkeypatch.setattr("app.services.analysis_service.shutil.which", lambda _: None)
    findings, warnings = await scan_code("eval(x)", "python")
    assert findings == []
    assert any("Ruff unavailable" in w for w in warnings)
    assert any("Semgrep unavailable" in w for w in warnings)


@pytest.mark.asyncio
async def test_no_code_execution(tmp_path):
    marker = tmp_path / "must-not-exist"
    await scan_code(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n", "python"
    )
    assert not marker.exists()


@pytest.mark.asyncio
async def test_semgrep_and_ruff_results(monkeypatch):
    import app.services.analysis_service as analysis

    monkeypatch.setattr(analysis.shutil, "which", lambda name: name)

    async def scanner(args, cwd):
        if args[0] == "ruff":
            return [
                {
                    "code": "F821",
                    "message": "Undefined name",
                    "location": {"row": 1},
                    "fix": None,
                }
            ]
        return {
            "results": [
                {
                    "check_id": "python-eval",
                    "extra": {
                        "severity": "WARNING",
                        "message": "eval needs validation",
                    },
                    "start": {"line": 1},
                }
            ],
            "errors": [{"message": "partial parse"}],
        }

    monkeypatch.setattr(analysis, "run_scanner", scanner)
    findings, warnings = await analysis.scan_code("eval(x)", "python")
    assert {f.source for f in findings} == {"ruff", "semgrep"}
    assert any("coverage is incomplete" in w for w in warnings)


@pytest.mark.asyncio
async def test_scanner_failure_warning(monkeypatch):
    import app.services.analysis_service as analysis

    monkeypatch.setattr(analysis.shutil, "which", lambda name: name)
    monkeypatch.setattr(analysis, "run_scanner", AsyncMock(side_effect=TimeoutError))
    findings, warnings = await analysis.scan_code("eval(x)", "python")
    assert findings == []
    assert sum("timed out" in w for w in warnings) == 2
