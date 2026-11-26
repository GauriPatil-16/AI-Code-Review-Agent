from unittest.mock import AsyncMock

import pytest
from fastmcp import Client

from app.mcp import mcp_server


@pytest.mark.asyncio
async def test_mcp_tools_over_protocol(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "review_pr_service", AsyncMock(return_value={"id": "test-review"})
    )
    monkeypatch.setattr(
        mcp_server.GroqService,
        "tests",
        AsyncMock(return_value="def test_example(): pass"),
    )
    async with Client(mcp_server.mcp) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools} == {
            "review_pull_request",
            "scan_file",
            "suggest_tests",
        }
        result = await client.call_tool(
            "scan_file", {"code": "x = 1", "language": "other"}
        )
        assert not result.is_error
        result = await client.call_tool("suggest_tests", {"code": "def f(): pass"})
        assert "test_example" in str(result)
        result = await client.call_tool(
            "review_pull_request", {"url": "https://github.com/a/b/pull/1"}
        )
        assert "test-review" in str(result)


@pytest.mark.asyncio
async def test_mcp_validation():
    async with Client(mcp_server.mcp) as client:
        with pytest.raises(Exception):
            await client.call_tool("scan_file", {"code": "", "language": "python"})
