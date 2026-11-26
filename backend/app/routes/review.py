import asyncio

from fastapi import APIRouter

from app.rag import rag_service
from app.schemas.schemas import (
    CodeRequest,
    IndexRequest,
    IndexResponse,
    PRRequest,
    ReviewResponse,
)
from app.services.groq_service import ServiceError
from app.services.review_service import review_code, review_pull_request

router = APIRouter(tags=["reviews"])


@router.post("/api/review/code", response_model=ReviewResponse)
@router.post("/run", response_model=ReviewResponse)
async def code_review(request: CodeRequest):
    return await review_code(request)


@router.post("/api/review/github", response_model=ReviewResponse)
async def github_review(request: PRRequest):
    return await review_pull_request(request)


@router.post("/api/rag/index", response_model=IndexResponse)
async def index_context(request: IndexRequest):
    if not rag_service.available():
        raise ServiceError("ChromaDB is not installed.", 503)
    try:
        count = await asyncio.to_thread(
            rag_service.index_documents,
            request.repository_id,
            [d.model_dump() for d in request.documents],
        )
        return {"indexed_chunks": count}
    except Exception:
        raise ServiceError(
            "Repository context indexing failed. Check backend storage permissions."
        ) from None
