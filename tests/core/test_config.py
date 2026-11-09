from finsight.core.config import Settings


def make_settings(database_url: str) -> Settings:
    return Settings(
        database_url=database_url,
        sec_user_agent="FinSight test@example.com",
        redis_url="redis://localhost:6379/0",
        celery_broker_url="redis://localhost:6379/1",
        celery_result_backend="redis://localhost:6379/1",
    )


def test_settings_adds_psycopg_driver_to_postgresql_url() -> None:
    settings = make_settings(
        "postgresql://user:password@host/database?sslmode=require"
    )

    assert settings.database_url == (
        "postgresql+psycopg://user:password@host/database?sslmode=require"
    )


def test_settings_adds_psycopg_driver_to_legacy_postgres_url() -> None:
    settings = make_settings("postgres://user:password@host/database")

    assert settings.database_url == "postgresql+psycopg://user:password@host/database"


def test_settings_preserves_explicit_sqlalchemy_driver() -> None:
    database_url = "postgresql+psycopg://user:password@host/database"

    assert make_settings(database_url).database_url == database_url
