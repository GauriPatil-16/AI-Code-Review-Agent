from fastapi import APIRouter

from app.schemas.schemas import ChatRequest, CodeRequest, TextResponse
from app.services.groq_service import GroqService

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


@router.post("/chat", response_model=TextResponse)
async def chat(request: ChatRequest):
    return {"content": await GroqService().chat(request)}


@router.post("/generate-tests", response_model=TextResponse)
async def generate_tests(request: CodeRequest):
    return {"content": await GroqService().tests(request)}
