import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import FixWorkflow, Review, ReviewSource
from app.schemas.schemas import ReviewResponse
from app.services.fix_service import delete_checkpoints
from app.services.review_service import list_reviews, serialize

router = APIRouter(prefix="/api/reviews", tags=["history"])


@router.get("", response_model=list[ReviewResponse])
def history(
    search: str = Query("", max_length=250),
    language: str = Query("", max_length=30),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    return list_reviews(db, search, language, limit, offset)


@router.get("/{review_id}", response_model=ReviewResponse)
def detail(review_id: str, db: Session = Depends(get_db)):
    row = db.get(Review, review_id)
    if row is None:
        raise HTTPException(404, "Review not found.")
    return serialize(row)


@router.delete("/{review_id}", status_code=204)
async def delete(review_id: str, db: Session = Depends(get_db)):
    def remove():
        row = db.get(Review, review_id)
        if row is None:
            raise HTTPException(404, "Review not found.")
        source = db.get(ReviewSource, review_id)
        workflow = db.scalar(
            select(FixWorkflow).where(FixWorkflow.review_id == review_id)
        )
        workflow_id = workflow.id if workflow else None
        if workflow:
            db.delete(workflow)
        if source:
            db.delete(source)
        db.delete(row)
        db.commit()
        return workflow_id

    workflow_id = await asyncio.to_thread(remove)
    if workflow_id:
        await delete_checkpoints(workflow_id)
    return Response(status_code=204)
