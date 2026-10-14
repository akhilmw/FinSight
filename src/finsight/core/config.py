from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FINSIGHT_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
