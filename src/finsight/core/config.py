from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FinSight API"
    environment: str = "development"
    database_url: str
    sec_user_agent: str
    sec_request_timeout_seconds: float = 20.0
    sec_requests_per_second: float = 5.0
    raw_document_directory: Path = Path("data/raw")
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 64
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    answer_model: str = "gpt-4o-mini"
    redis_url: str
    answer_cache_ttl_seconds: int = 86400
    celery_broker_url: str
    celery_result_backend: str
    mlflow_tracing_enabled: bool = False
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_name: str = "finsight"

    @field_validator("database_url", mode="before")
    @classmethod
    def use_psycopg_driver(cls, value: object) -> object:
        """Use SQLAlchemy's psycopg 3 dialect for provider connection URLs."""
        if not isinstance(value, str):
            return value

        for prefix in ("postgresql://", "postgres://"):
            if value.startswith(prefix):
                return f"postgresql+psycopg://{value.removeprefix(prefix)}"

        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FINSIGHT_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
