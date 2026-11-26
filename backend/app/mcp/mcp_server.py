from fastmcp import FastMCP

from app.database.db import init_db
from app.schemas.schemas import CodeRequest, PRRequest
from app.services.analysis_service import scan_code
from app.services.groq_service import GroqService
from app.services.review_service import review_pull_request as review_pr_service

mcp = FastMCP("AI Code Review Agent")


async def review_pull_request_impl(url: str) -> dict:
    init_db()
    return await review_pr_service(PRRequest(url=url))


async def scan_file_impl(code: str, language: str = "python") -> dict:
    request = CodeRequest(code=code, language=language)
    findings, warnings = await scan_code(request.code, request.language)
    return {"findings": [f.model_dump() for f in findings], "warnings": warnings}


async def suggest_tests_impl(code: str, language: str = "python") -> dict:
    return {
        "content": await GroqService().tests(CodeRequest(code=code, language=language))
    }


@mcp.tool
async def review_pull_request(url: str) -> dict:
    """Review a GitHub PR, persist findings and return results. Does not post comments."""
    return await review_pull_request_impl(url)


@mcp.tool
async def scan_file(code: str, language: str = "python") -> dict:
    """Run safe static analysis on source text; never execute the submitted code."""
    return await scan_file_impl(code, language)


@mcp.tool
async def suggest_tests(code: str, language: str = "python") -> dict:
    """Generate unit tests using Groq. Generated tests are not executed."""
    return await suggest_tests_impl(code, language)


if __name__ == "__main__":
    mcp.run(transport="stdio")
