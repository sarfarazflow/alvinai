from celery import Celery
from app.core.config import get_settings

settings = get_settings()

celery = Celery(
    "alvinai",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
)

celery.autodiscover_tasks(["app.ingestion"])
