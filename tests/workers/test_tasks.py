import asyncio
from typing import Any, cast

import pytest

from finsight.workers import tasks


def test_process_filing_task_runs_async_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    expected = {
        "filing_id": 3,
        "document_path": "data/raw/filing.html",
        "sections_created": 12,
        "chunks_created": 80,
        "chunks_embedded": 80,
    }

    async def fake_run_filing_pipeline(
        filing_id: int,
    ) -> dict[str, int | str]:
        await asyncio.sleep(0)
        calls.append(filing_id)
        return expected

    monkeypatch.setattr(
        tasks,
        "run_filing_pipeline",
        fake_run_filing_pipeline,
    )

    task = cast(Any, tasks.process_filing_task)
    result = task.run(3)

    assert result == expected
    assert calls == [3]
    assert task.acks_late is True
    assert task.reject_on_worker_lost is True
