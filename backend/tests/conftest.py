import os
import tempfile

import pytest
from fastapi.testclient import TestClient

# Set configuration before importing the application. Production databases stay untouched.
_test_data = tempfile.TemporaryDirectory(
    prefix="review-tests-", ignore_cleanup_errors=True
)
os.environ["DATABASE_URL"] = (
    "sqlite:///" + _test_data.name.replace("\\", "/") + "/reviews.db"
)
os.environ["CHROMA_PATH"] = _test_data.name + "/chroma"
os.environ["GROQ_API_KEY"] = ""
os.environ["FIX_WORKER_ENABLED"] = "false"

from app.main import app  # noqa: E402
from app.database.db import Base, engine  # noqa: E402
from app.schemas.schemas import AIReview  # noqa: E402


@pytest.fixture
def client():
    Base.metadata.drop_all(engine)
    with TestClient(app) as instance:
        yield instance


@pytest.fixture
def ai_result():
    return AIReview(
        summary="The divisor can be zero.",
        improved_code='def divide(a, b):\n    if b == 0: raise ValueError("zero")\n    return a / b',
        findings=[
            {
                "title": "Zero divisor",
                "severity": "medium",
                "category": "bug",
                "description": "A zero divisor raises ZeroDivisionError.",
                "suggestion": "Validate the divisor.",
                "file": "snippet",
                "line": 2,
                "source": "groq",
            }
        ],
    )


@pytest.fixture
def mock_review(monkeypatch, ai_result):
    from app.services.groq_service import GroqService
    from app.agents import review_agent

    async def review(self, payload):
        return ai_result

    async def scan(*args):
        return [], ["Test scanner stub; no real scan."]

    monkeypatch.setattr(GroqService, "review", review)
    monkeypatch.setattr(review_agent, "scan_code", scan)
    return ai_result
