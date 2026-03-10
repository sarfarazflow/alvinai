from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me-in-production"
    ALLOWED_HOSTS: str = "localhost,127.0.0.1"

    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "alvinai"
    POSTGRES_USER: str = "alvinai"
    POSTGRES_PASSWORD: str = ""

    REDIS_URL: str = "redis://redis:6379/0"

    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET_DOCUMENTS: str = "documents"

    VLLM_BASE_URL: str = "http://vllm_blue:8080/v1"
    VLLM_MODEL_ID: str = ""
    VLLM_MODEL_NAME: str = "alvinai-v1"
    VLLM_TIMEOUT_SECONDS: int = 60
    RUNPOD_API_KEY: str = ""

    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_MODEL: str = "mistral:7b"

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.72
    RAG_COMPLIANCE_SIMILARITY_THRESHOLD: float = 0.80
    RAG_CACHE_TTL: int = 3600

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def is_runpod_serverless(self) -> bool:
        """True only for RunPod serverless (needs API key + /run endpoint)."""
        return bool(self.RUNPOD_API_KEY) and "runpod" in self.VLLM_BASE_URL.lower()

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
