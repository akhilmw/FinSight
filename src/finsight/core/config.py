from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FinSight API"
    environment: str = "development"
    database_url: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FINSIGHT_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()