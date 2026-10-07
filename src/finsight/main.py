from fastapi import FastAPI

from finsight.api.routes.health import router as health_router
from finsight.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Evidence-backed research assistant for SEC filings",
)
app.include_router(health_router)
