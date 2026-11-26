from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

Language = Literal[
    "python", "javascript", "typescript", "java", "c++", "other", "multiple"
]
Action = Literal[
    "Explain Code",
    "Find Bugs",
    "Fix Code",
    "Optimize Code",
    "Refactor Code",
    "Generate Unit Tests",
    "Security Review",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextDocument(StrictModel):
    path: str = Field(min_length=1, max_length=250)
    content: str = Field(min_length=1, max_length=20000)


class CodeRequest(StrictModel):
    code: str = Field(min_length=1, max_length=60000)
    language: Language = "python"
    instructions: str = Field(default="", max_length=2000)
    repository_id: str | None = Field(default=None, max_length=200)
    documents: list[ContextDocument] = Field(default_factory=list, max_length=10)

    @field_validator("code")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Code must contain text without null bytes.")
        return value


class PRRequest(StrictModel):
    url: str = Field(max_length=500)
    instructions: str = Field(default="", max_length=2000)

    @field_validator("url")
    @classmethod
    def github_pr(cls, value: str) -> str:
        import re

        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "github.com"
            or parsed.query
            or parsed.fragment
            or not re.fullmatch(
                r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/[1-9][0-9]*/?", parsed.path
            )
        ):
            raise ValueError(
                "Use https://github.com/owner/repository/pull/123 without query parameters."
            )
        return value.rstrip("/")


class Finding(StrictModel):
    title: str = Field(min_length=1, max_length=250)
    severity: Literal["critical", "high", "medium", "low"]
    category: Literal["bug", "security", "performance", "quality"]
    description: str = Field(min_length=1, max_length=5000)
    suggestion: str = Field(min_length=1, max_length=5000)
    file: str | None = None
    line: int | None = Field(default=None, ge=1)
    source: Literal["groq", "ruff", "semgrep"] = "groq"


class AIReview(StrictModel):
    summary: str = Field(min_length=1, max_length=10000)
    findings: list[Finding] = Field(max_length=100)
    improved_code: str = Field(max_length=100000)


class ReviewResponse(AIReview):
    id: str
    title: str
    language: str
    review_type: str
    source_reference: str | None
    original_code: str
    created_at: datetime
    warnings: list[str]
    context_sources: list[str]


class ChatMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(StrictModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=20)
    action: Action = "Explain Code"

    @field_validator("messages")
    @classmethod
    def user_last(cls, value):
        if value[-1].role != "user" or sum(len(m.content) for m in value) > 80000:
            raise ValueError(
                "End with a user message; conversation limit is 80,000 characters."
            )
        return value


class TextResponse(StrictModel):
    content: str


class FixedFile(StrictModel):
    path: str = Field(min_length=1, max_length=500)
    code: str = Field(max_length=100000)
    issue_ids: list[int] = Field(min_length=1, max_length=100)
    explanation: str = Field(min_length=1, max_length=4000)


class SkippedIssue(StrictModel):
    issue_id: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=2000)


class FixPlan(StrictModel):
    files: list[FixedFile] = Field(max_length=8)
    skipped_issues: list[SkippedIssue] = Field(max_length=100)


class IssueVerification(StrictModel):
    issue_id: int = Field(ge=0)
    verdict: Literal["fixed", "remaining", "unsupported", "uncertain"]
    evidence: str = Field(min_length=1, max_length=3000)


class FixVerification(StrictModel):
    issues: list[IssueVerification] = Field(max_length=100)
    behavior_preserved: bool
    unintended_changes: list[str] = Field(max_length=50)


class FixDecision(StrictModel):
    decision: Literal["approve", "reject"]
    proposal_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class FixWorkflowResponse(StrictModel):
    id: str
    review_id: str
    status: str
    proposal: dict
    proposal_hash: str | None
    decision: str | None
    branch: str | None
    commit_sha: str | None
    pr_url: str | None
    error: str | None


class IndexRequest(StrictModel):
    repository_id: str = Field(min_length=1, max_length=200)
    documents: list[ContextDocument] = Field(min_length=1, max_length=50)


class IndexResponse(StrictModel):
    indexed_chunks: int


class StatsResponse(StrictModel):
    total_reviews: int
    bugs: int
    security: int
    improvements: int
    severity: dict[str, int]


class ActivityPoint(StrictModel):
    date: str
    label: str
    reviews: int


class StatusResponse(StrictModel):
    provider: str = "Groq"
    model: str
    groq_connected: bool
    groq_status: str
    github_configured: bool
    ruff: bool
    semgrep: bool
    rag_available: bool
