from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, JSON, String, Text, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.database.db import Base


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid4())
    )
    title: Mapped[str] = mapped_column(String(300))
    language: Mapped[str] = mapped_column(String(30))
    review_type: Mapped[str] = mapped_column(String(20))
    source_reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str] = mapped_column(Text)
    original_code: Mapped[str] = mapped_column(Text)
    improved_code: Mapped[str] = mapped_column(Text)
    findings: Mapped[list] = mapped_column(JSON)
    warnings: Mapped[list] = mapped_column(JSON)
    context_sources: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class ReviewSource(Base):
    """Immutable inputs retained separately to avoid changing existing review rows."""
    __tablename__ = "review_sources"
    review_id: Mapped[str] = mapped_column(String, primary_key=True)
    snapshot: Mapped[dict] = mapped_column(JSON)


class FixWorkflow(Base):
    __tablename__ = "fix_workflows"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    review_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    status: Mapped[str] = mapped_column(String, default="Analyzing")
    proposal: Mapped[dict] = mapped_column(JSON, default=dict)
    proposal_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    decision: Mapped[str | None] = mapped_column(String, nullable=True)
    approved_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    approved_at: Mapped[str | None] = mapped_column(String, nullable=True)
    branch: Mapped[str | None] = mapped_column(String, nullable=True)
    commit_sha: Mapped[str | None] = mapped_column(String, nullable=True)
    git_identity: Mapped[dict] = mapped_column(JSON, default=dict)
    pr_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    lease_until: Mapped[float] = mapped_column(Float, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String, nullable=True)
    finished: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
