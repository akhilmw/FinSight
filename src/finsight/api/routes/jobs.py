from typing import Annotated, Any

from celery import Celery, states
from fastapi import APIRouter, Depends

from finsight.api.dependencies import get_celery_app
from finsight.schemas import JobStatusResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])

CeleryDependency = Annotated[Celery, Depends(get_celery_app)]


@router.get("/{task_id}", response_model=JobStatusResponse)
def get_job_status(
    task_id: str,
    task_queue: CeleryDependency,
) -> JobStatusResponse:
    task = task_queue.AsyncResult(task_id)
    task_status = task.state.lower()
    result: dict[str, Any] | None = None
    error: str | None = None

    if task.state == states.SUCCESS and isinstance(task.result, dict):
        result = task.result
    elif task.state == states.FAILURE:
        error = str(task.result)

    successful = task.successful() if task.ready() else None
    return JobStatusResponse(
        task_id=task_id,
        status=task_status,
        ready=task.ready(),
        successful=successful,
        result=result,
        error=error,
    )
