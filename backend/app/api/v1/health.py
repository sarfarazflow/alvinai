from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "env": settings.APP_ENV,
        "model": settings.VLLM_MODEL_NAME,
        "inference": "runpod-serverless" if settings.is_runpod_serverless else "vllm-direct",
    }
