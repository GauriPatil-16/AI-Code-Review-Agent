import asyncio
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.rag import rag_service
from app.schemas.schemas import AIReview, CodeRequest
from app.services.analysis_service import scan_code
from app.services.groq_service import GroqService, ServiceError
from app.services.review_service import save_review


class ReviewState(TypedDict, total=False):
    code: str
    language: str
    instructions: str
    title: str
    review_type: str
    source_reference: str | None
    repository_id: str | None
    files: list[dict]
    documents: list[dict]
    warnings: list[str]
    context: list[dict]
    context_sources: list[str]
    static_findings: list[Any]
    findings: list[Any]
    ai: AIReview
    result: dict
    repository: str
    source: dict


def validate_code(state):
    if not state.get("files") or sum(len(f["code"]) for f in state["files"]) > 60000:
        raise ServiceError("Source exceeds review limits or is empty.", 422)
    for file in state["files"]:
        CodeRequest(code=file["code"], language=file["language"])
    return {"warnings": list(state.get("warnings", []))}


def fetch_context(state):
    # GitHub fetches immutable head-commit source before entering the shared graph.
    return {"context": [], "context_sources": []}


async def static_analysis(state):
    # Bound concurrency to two subprocess-producing scans.
    semaphore = asyncio.Semaphore(2)

    async def scan(file):
        async with semaphore:
            return await scan_code(file["code"], file["language"], file["path"])

    results = await asyncio.gather(*(scan(f) for f in state["files"]))
    return {
        "static_findings": [f for findings, _ in results for f in findings],
        "warnings": state["warnings"]
        + [w for _, warnings in results for w in warnings],
    }


async def retrieve_context(state):
    repository_id = state.get("repository_id")
    documents = state.get("documents", [])
    if not repository_id:
        return {
            "warnings": state["warnings"]
            + ["RAG skipped: no repository context provided."]
        }
    if not rag_service.available():
        return {
            "warnings": state["warnings"]
            + ["RAG unavailable: ChromaDB is not installed."]
        }
    try:
        if documents or state["review_type"] == "github":
            source = documents + (
                [{"path": f["path"], "content": f["code"]} for f in state["files"]]
                if state["review_type"] == "github"
                else []
            )
            await asyncio.to_thread(rag_service.index_documents, repository_id, source)
        context = await asyncio.to_thread(
            rag_service.retrieve,
            repository_id,
            state["code"][:12000] + state.get("instructions", ""),
        )
        return {
            "context": context,
            "context_sources": [c["source"] for c in context],
            "warnings": state["warnings"]
            + ([] if context else ["RAG found no relevant indexed context."]),
        }
    except Exception:
        return {
            "warnings": state["warnings"]
            + ["RAG retrieval failed; review continued without retrieved context."]
        }


async def generate_review(state):
    result = await GroqService().review(
        {
            "language": state["language"],
            "source_kind": state["review_type"],
            "files": state["files"],
            "custom_instructions": state.get("instructions", ""),
            "retrieved_context": state.get("context", []),
            "scanner_findings": [f.model_dump() for f in state["static_findings"]],
        }
    )
    return {"ai": result}


def validate_findings(state):
    ai = AIReview.model_validate(state["ai"])
    file_map = {f["path"]: f["code"] for f in state["files"]}
    warnings, combined, seen = list(state["warnings"]), [], set()
    for finding in [
        *state["static_findings"],
        *[f.model_copy(update={"source": "groq"}) for f in ai.findings],
    ]:
        if finding.file not in file_map:
            if len(file_map) == 1 and finding.file in (None, "snippet"):
                finding = finding.model_copy(update={"file": next(iter(file_map))})
            else:
                warnings.append(
                    "An AI finding referred to an unknown file; its location was removed."
                )
                finding = finding.model_copy(update={"file": None, "line": None})
        if finding.line and (
            not finding.file or finding.line > len(file_map[finding.file].splitlines())
        ):
            warnings.append("An out-of-range finding line was removed.")
            finding = finding.model_copy(update={"line": None})
        key = (finding.file, finding.line, finding.title.casefold())
        if key not in seen:
            seen.add(key)
            combined.append(finding)
    if state["review_type"] == "github":
        ai = ai.model_copy(update={"improved_code": ""})
    return {"ai": ai, "findings": combined, "warnings": warnings}


def persist(state):
    return {"result": save_review(state)}


def queue_fixes(state):
    # Persist the job; the lifespan worker executes it independently of the
    # review response and can recover it after a process restart.
    from app.services.fix_service import enqueue_fix
    if state.get("findings"):
        enqueue_fix(state["result"]["id"])
    return {}


def build_graph():
    graph = StateGraph(ReviewState)
    nodes = [
        ("validate_code", validate_code),
        ("fetch_context", fetch_context),
        ("static_analysis", static_analysis),
        ("retrieve_context", retrieve_context),
        ("generate_review", generate_review),
        ("validate_findings", validate_findings),
        ("save_review", persist),
        ("queue_fixes", queue_fixes),
    ]
    previous = START
    for name, function in nodes:
        graph.add_node(name, function)
        graph.add_edge(previous, name)
        previous = name
    graph.add_edge(previous, END)
    return graph.compile()


review_graph = build_graph()


async def run_review(state: dict) -> dict:
    return (await review_graph.ainvoke(state, config={"recursion_limit": 12}))["result"]
