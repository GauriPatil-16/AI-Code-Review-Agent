from contextlib import asynccontextmanager
from contextlib import suppress
import asyncio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.database.db import init_db
from app.routes import assistant, dashboard, github, history, review, fixes
from app.services.fix_service import worker
from app.services.groq_service import ServiceError


@asynccontextmanager
async def lifespan(app):
    init_db()
    task = asyncio.create_task(worker()) if get_settings().fix_worker_enabled else None
    try:
        yield
    finally:
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


app = FastAPI(title="AI Code Review Agent", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def request_limit(request: Request, call_next):
    # Bound streamed and chunked bodies as well as Content-Length requests.
    if request.method in ("POST", "PUT", "PATCH"):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 1_500_000:
                return JSONResponse(
                    status_code=413, content={"detail": "Request body exceeds 1.5 MB."}
                )
        request._body = bytes(body)
    return await call_next(request)


@app.exception_handler(ServiceError)
async def service_error(request: Request, error: ServiceError):
    return JSONResponse(status_code=error.status_code, content={"detail": str(error)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


for router in (
    review.router,
    assistant.router,
    github.router,
    history.router,
    dashboard.router,
    fixes.router,
):
    app.include_router(router)
