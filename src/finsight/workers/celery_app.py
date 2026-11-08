from celery import Celery

from finsight.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "finsight",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["finsight.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3_600,
    broker_connection_retry_on_startup=True,
)