import asyncio
import shutil

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.db import get_db
from app.rag.rag_service import available
from app.schemas.schemas import ActivityPoint, StatsResponse, StatusResponse
from app.services.groq_service import GroqService
from app.services.review_service import activity, statistics

router = APIRouter(tags=["dashboard", "settings"])


@router.get("/api/dashboard/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_db)):
    return statistics(db)


@router.get("/api/dashboard/activity", response_model=list[ActivityPoint])
def review_activity(db: Session = Depends(get_db)):
    return activity(db)


@router.get("/api/settings/status", response_model=StatusResponse)
async def status():
    try:
        connected, description = await asyncio.wait_for(
            GroqService().connection_status(), timeout=10
        )
    except TimeoutError:
        connected, description = False, "Connection check timed out"
    return {
        "model": get_settings().groq_model,
        "groq_connected": connected,
        "groq_status": description,
        "github_configured": bool(get_settings().github_token),
        "ruff": bool(shutil.which("ruff")),
        "semgrep": bool(shutil.which("semgrep")),
        "rag_available": available(),
    }
