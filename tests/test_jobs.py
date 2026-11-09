from collections.abc import Iterator
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from celery import states
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finsight.api.dependencies import get_celery_app
from finsight.db.base import Base
from finsight.db.models import Filing
from finsight.db.session import get_db
from finsight.main import app


@pytest.fixture
def job_client() -> Iterator[tuple[TestClient, MagicMock]]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(bind=engine)

    with testing_session() as session:
        session.add(
            Filing(
                accession_number="0001045810-26-000021",
                cik="0001045810",
                company_name="NVIDIA CORP",
                form_type="10-K",
                filing_date=date(2026, 2, 25),
                source_url="https://www.sec.gov/example.htm",
            )
        )
        session.commit()

    def override_get_db() -> Iterator[Session]:
        with testing_session() as session:
            yield session

    task_queue = MagicMock()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_celery_app] = lambda: task_queue

    with TestClient(app) as client:
        yield client, task_queue

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


def test_enqueue_filing_pipeline_returns_task_id(
    job_client: tuple[TestClient, MagicMock],
) -> None:
    client, task_queue = job_client
    task_queue.send_task.return_value = SimpleNamespace(id="task-123")

    response = client.post("/filings/1/pipeline")

    assert response.status_code == 202
    assert response.json() == {
        "task_id": "task-123",
        "filing_id": 1,
        "status": "queued",
    }
    task_queue.send_task.assert_called_once_with(
        "finsight.process_filing",
        args=[1],
    )


def test_enqueue_filing_pipeline_rejects_unknown_filing(
    job_client: tuple[TestClient, MagicMock],
) -> None:
    client, task_queue = job_client

    response = client.post("/filings/999/pipeline")

    assert response.status_code == 404
    task_queue.send_task.assert_not_called()


@pytest.mark.parametrize(
    ("state", "value", "ready", "successful", "expected"),
    [
        (states.PENDING, None, False, False, (None, None, None)),
        (
            states.SUCCESS,
            {"filing_id": 1, "chunks_embedded": 42},
            True,
            True,
            (
                True,
                {"filing_id": 1, "chunks_embedded": 42},
                None,
            ),
        ),
        (
            states.FAILURE,
            RuntimeError("embedding failed"),
            True,
            False,
            (False, None, "embedding failed"),
        ),
    ],
)
def test_get_job_status(
    job_client: tuple[TestClient, MagicMock],
    state: str,
    value: object,
    ready: bool,
    successful: bool,
    expected: tuple[bool | None, dict[str, object] | None, str | None],
) -> None:
    client, task_queue = job_client
    task = MagicMock(state=state, result=value)
    task.ready.return_value = ready
    task.successful.return_value = successful
    task_queue.AsyncResult.return_value = task

    response = client.get("/jobs/task-123")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "task-123"
    assert body["status"] == state.lower()
    assert (body["successful"], body["result"], body["error"]) == expected
