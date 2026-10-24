from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from redis.asyncio import Redis

from finsight.api.routes.answers import router as answers_router
from finsight.api.routes.filings import router as filings_router
from finsight.api.routes.health import router as health_router
from finsight.api.routes.search import router as search_router
from finsight.core.config import get_settings

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    redis_client = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )

    app.state.redis_client = redis_client

    try:
        yield
    finally:
        await redis_client.aclose()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Evidence-backed research assistant for SEC filings",
    lifespan=lifespan,
)
app.include_router(health_router)
app.include_router(filings_router)
app.include_router(search_router)
app.include_router(answers_router)
