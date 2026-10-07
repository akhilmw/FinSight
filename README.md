# FinSight

FinSight is an evidence-backed research assistant for SEC company filings.

## Day 1

The initial milestone is a small FastAPI service with a health endpoint. Database and Docker
setup are intentionally left as hands-on exercises.

## Local development

```bash
uv sync --dev
uv run uvicorn finsight.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI.

## Checks

```bash
uv run ruff check .
uv run pyright
uv run pytest
```

