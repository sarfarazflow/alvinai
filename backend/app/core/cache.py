import json
import redis.asyncio as redis
from app.core.config import get_settings

settings = get_settings()

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


async def cache_get(key: str) -> dict | None:
    data = await redis_client.get(key)
    if data:
        return json.loads(data)
    return None


async def cache_set(key: str, value: dict, ttl: int | None = None) -> None:
    ttl = ttl or settings.RAG_CACHE_TTL
    await redis_client.set(key, json.dumps(value), ex=ttl)


async def cache_delete(key: str) -> None:
    await redis_client.delete(key)
