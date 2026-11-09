# FinSight

FinSight is an evidence-grounded research assistant for SEC filings. It ingests
10-K and 10-Q filings from EDGAR, converts them into searchable sections and
chunks, and answers financial research questions with citations back to the
original filing.

The project is built as a production-oriented AI engineering system rather than
a standalone chatbot: it includes a reproducible ingestion pipeline, hybrid
retrieval, reranking, retrieval evaluation, structured answer generation,
streaming, caching, and background workers.

## Highlights

- End-to-end SEC EDGAR ingestion from ticker discovery to locally archived HTML
- Structured pipeline: `filing -> sections -> chunks -> embeddings`
- Vector search with pgvector and PostgreSQL full-text keyword search
- Reciprocal-rank fusion (RRF) plus cross-encoder reranking
- Metadata filtering by company, filing type, year, and filing ID
- Evidence-grounded answers with validated chunk-level citations
- Explicit insufficient-evidence behavior to reduce unsupported answers
- Server-Sent Events (SSE) for progressive answer streaming
- Redis cache-aside layer with deterministic request keys and graceful fallback
- Celery worker for durable, asynchronous filing processing
- MLflow traces with model latency, token usage, and estimated cost
- Adversarial evidence filtering for document prompt-injection attempts
- Measured retrieval quality, latency, and indexing performance
- 170 automated tests across ingestion, retrieval, generation, evaluation, API,
  cache, observability, security, and worker layers

## Architecture

```mermaid
flowchart LR
    SEC[SEC EDGAR APIs] --> DISCOVER[Discover filing metadata]
    DISCOVER --> PG[(PostgreSQL)]
    SEC --> DOWNLOAD[Download filing HTML]
    DOWNLOAD --> RAW[(Local raw HTML)]
    RAW --> PARSE[Parse and extract sections]
    PARSE --> CHUNK[Token-aware chunking]
    CHUNK --> EMBED[OpenAI embeddings]
    EMBED --> VECTOR[(pgvector + HNSW)]
    CHUNK --> FTS[(PostgreSQL full-text index)]

    REDIS_QUEUE[(Redis broker)] --> CELERY[Celery worker]
    CELERY --> DOWNLOAD

    USER[Client] --> API[FastAPI]
    API --> CACHE{Redis answer cache}
    CACHE -->|miss| SEARCH[Hybrid retrieval]
    VECTOR --> SEARCH
    FTS --> SEARCH
    SEARCH --> RRF[Reciprocal-rank fusion]
    RRF --> RERANK[Cross-encoder reranking]
    RERANK --> LLM[Structured answer generation]
    LLM --> MLFLOW[(MLflow traces)]
    LLM --> VALIDATE[Citation validation]
    VALIDATE --> API
    API -->|JSON or SSE| USER
```

PostgreSQL is the system of record for filing metadata, extracted sections,
chunks, and embeddings. Redis serves three separate roles: answer cache, Celery
broker, and Celery result backend.

## Retrieval results

The current evaluation set contains 50 manually labeled questions and 60
relevant-chunk labels from NVIDIA's FY2026 10-K. It includes exact-term,
semantic, and mixed queries.

| Retrieval mode | Recall@5 | Recall@10 | MRR | Mean latency |
| --- | ---: | ---: | ---: | ---: |
| Vector | 0.570 | 0.630 | 0.428 | 410.8 ms |
| Keyword | 0.650 | 0.790 | 0.453 | 115.7 ms |
| Hybrid | 0.630 | 0.760 | 0.523 | 519.3 ms |
| **Hybrid + reranking** | **0.870** | **0.970** | **0.755** | 1425.3 ms |

Cross-encoder reranking improves Recall@10 from 0.76 to 0.97 and MRR from
0.523 to 0.755 over the hybrid baseline. The tradeoff is higher latency, which
is why FinSight exposes each retrieval mode independently.

### Answer quality and security

The same 50 questions were run end to end through retrieval, reranking, answer
generation, citation validation, and a separate structured LLM judge.

| Metric | Result |
| --- | ---: |
| Sufficient-evidence rate | 1.000 |
| Mean groundedness | **0.962** |
| Mean citation correctness | **0.947** |
| Mean end-to-end latency | 8704.4 ms |
| OpenAI traces | 150 |
| Total tokens | 252,759 |
| Estimated model cost | **$0.0449** |

FinSight also runs five adversarial cases containing direct system overrides,
forced outputs, fabricated facts, and instruction-exfiltration attempts. The
first run passed only 1/5 cases and exposed a real weakness. After adding a
deterministic suspicious-evidence filter and stronger evidence boundaries, the
unchanged suite passed **5/5**. Both evaluation outputs are saved under
`evals/results/`.

See the [answer quality and security report](evals/answer_report.md) for the
methodology, cost breakdown, adversarial results, and limitations.

See the [full retrieval evaluation](evals/retrieval_report.md) for methodology,
per-mode results, and limitations.

### Indexing benchmark

Measured locally on one NVIDIA 10-K:

| Metric | Result |
| --- | ---: |
| Extracted sections | 23 |
| Generated chunks | 162 |
| Embedding batches | 3 |
| Processing time | 350.2 ms |
| Chunking time | 580.6 ms |
| Embedding and storage time | 5089.1 ms |
| **Total indexing time** | **6019.9 ms** |
| Throughput | 26.9 chunks/s |

These are local, single-run measurements and should be treated as an engineering
baseline rather than a production benchmark.

## How retrieval works

FinSight supports four retrieval configurations:

| Mode | Behavior | Best suited for |
| --- | --- | --- |
| `vector` | Cosine similarity over OpenAI embeddings | Paraphrases and semantic questions |
| `keyword` | PostgreSQL full-text search | Exact financial terms, names, and phrases |
| `hybrid` | Fuses vector and keyword rankings with RRF | General-purpose retrieval |
| `hybrid_rerank` | Reranks hybrid candidates with a cross-encoder | Highest retrieval quality |

The chunking strategy respects filing section boundaries, targets a maximum of
600 tokens, and uses a 100-token overlap. Embeddings are stored in a
1536-dimensional pgvector column and searched through an HNSW cosine index.

For answer generation, retrieved chunks are treated as untrusted evidence. The
model is instructed not to follow instructions found inside filings, and every
returned citation is checked against the retrieved chunk IDs. If the evidence
cannot support an answer, the API returns `sufficient_evidence: false` with no
citations.

## Tech stack

| Area | Technology |
| --- | --- |
| API and validation | FastAPI, Pydantic |
| Database | PostgreSQL 16, SQLAlchemy 2, Alembic |
| Vector store | pgvector with HNSW indexing |
| Keyword retrieval | PostgreSQL full-text search with a GIN index |
| Embeddings | OpenAI `text-embedding-3-small` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer generation | OpenAI `gpt-4o-mini` with Structured Outputs |
| Parsing and chunking | Beautiful Soup, lxml, tiktoken |
| Cache and queue | Redis, Celery |
| Observability | MLflow OpenAI tracing |
| HTTP and streaming | HTTPX, Server-Sent Events |
| Tooling | uv, Ruff, Pyright, pytest, Docker Compose |

## Data model

```text
Filing
  └── FilingSection (one filing has many sections)
        └── FilingChunk (one section has many chunks)
              └── embedding (pgvector)
```

- `Filing` stores SEC metadata such as CIK, accession number, form type,
  reporting period, filing date, source URL, local content path, and pipeline
  status.
- `FilingSection` stores the SEC item, title, extracted text, and source line
  range.
- `FilingChunk` stores ordered chunk text, token offsets, token count, and its
  embedding.

## Local setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker with Docker Compose
- An OpenAI API key
- A descriptive SEC user agent containing a contact email

### 1. Configure the environment

```bash
cp .env.example .env
```

At minimum, replace these values in `.env`:

```dotenv
FINSIGHT_SEC_USER_AGENT=FinSight your-email@example.com
FINSIGHT_OPENAI_API_KEY=your_openai_api_key
POSTGRES_PASSWORD=choose_a_password
FINSIGHT_DATABASE_URL=postgresql+psycopg://finsight:choose_a_password@localhost:5432/finsight
```

### 2. Install dependencies and start infrastructure

```bash
uv sync --dev
docker compose up -d
uv run alembic upgrade head
```

Confirm PostgreSQL and Redis are healthy:

```bash
docker compose ps
```

### 3. Start the API

```bash
uv run uvicorn finsight.main:app --reload
```

The interactive API documentation is available at
<http://127.0.0.1:8000/docs>.

### 4. Start the background worker

For local development on macOS:

```bash
uv run celery -A finsight.workers.celery_app:celery_app worker --loglevel=INFO --pool=solo
```

On Linux, Celery's default prefork pool can process multiple filings
concurrently.

### 5. Enable MLflow tracing (optional)

Start a local tracking server:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
```

Then set `FINSIGHT_MLFLOW_TRACING_ENABLED=true` in `.env`. The MLflow UI is
available at <http://127.0.0.1:5000> and records prompts, responses, streaming
events, latency, token usage, and estimated model cost.

## API walkthrough

### Discover and store recent filings

```bash
curl -X POST http://127.0.0.1:8000/filings/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "NVDA",
    "form_types": ["10-K", "10-Q"],
    "limit": 5
  }'
```

The response reports how many filings were discovered, created, or skipped.
Repeated ingestion is idempotent because accession numbers are unique.

### Run the synchronous indexing pipeline

Use the filing ID returned by ingestion:

```bash
curl -X POST http://127.0.0.1:8000/filings/3/download
curl -X POST http://127.0.0.1:8000/filings/3/process
curl -X POST http://127.0.0.1:8000/filings/3/chunk
curl -X POST http://127.0.0.1:8000/filings/3/embed
```

The full pipeline can be submitted through the API while the worker is running:

```bash
curl -X POST http://127.0.0.1:8000/filings/3/pipeline
```

The response contains a task ID. Poll its status with:

```bash
curl http://127.0.0.1:8000/jobs/YOUR_TASK_ID
```

### Search indexed filings

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What factors could negatively affect revenue?",
    "mode": "hybrid_rerank",
    "limit": 5,
    "filing_id": 3
  }'
```

### Generate a cited answer

```bash
curl -X POST http://127.0.0.1:8000/answers \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What factors could negatively affect NVIDIA revenue?",
    "filing_id": 3,
    "evidence_limit": 5
  }'
```

The response contains the answer, an evidence-sufficiency decision, and source
metadata for every cited chunk.

### Stream an answer with SSE

```bash
curl -N -X POST http://127.0.0.1:8000/answers/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize NVIDIA revenue risks.",
    "filing_id": 3,
    "evidence_limit": 5
  }'
```

The stream emits `retrieval`, `answer_delta`, `citations`, `done`, and `error`
events. Cached answers use the same event contract.

## Caching and background processing

Generated answers use a cache-aside strategy. FinSight normalizes the request
and model name, hashes that payload with SHA-256, and stores the complete
structured response in Redis. The default TTL is 24 hours.

Redis failures do not break answer generation: cache reads and writes fail open,
so PostgreSQL, retrieval, and the model remain the source of truth.

The Celery worker runs the download, parse, chunk, and embed stages outside the
API process. Each task opens its own database session and updates filing status
through the pipeline, allowing long-running ingestion work to survive beyond an
individual HTTP request.

## Evaluation and quality checks

Reproduce the retrieval comparison:

```bash
uv run python scripts/evaluate_retrieval.py
```

Evaluate groundedness, citations, latency, tokens, and cost:

```bash
uv run python scripts/evaluate_answers.py
uv run python scripts/summarize_mlflow_traces.py
```

Run the adversarial document suite:

```bash
uv run python scripts/evaluate_prompt_injection.py
```

Benchmark the indexing pipeline for an existing filing:

```bash
uv run python scripts/benchmark_indexing.py 3
```

Run the project checks:

```bash
uv run ruff check .
uv run pyright
uv run pytest -q
```

## Repository layout

```text
FinSight/
├── alembic/                 # Database migrations
├── data/raw/                # Downloaded filing HTML (gitignored)
├── docker/postgres/         # PostgreSQL and pgvector initialization
├── evals/                   # Retrieval dataset, reports, and saved results
├── scripts/                 # Ingestion, inspection, evaluation, and benchmarks
├── src/finsight/
│   ├── api/                 # FastAPI routes and schemas
│   ├── clients/             # SEC EDGAR client
│   ├── core/                # Configuration
│   ├── db/                  # SQLAlchemy models and sessions
│   ├── evaluation/          # Retrieval, answer, and security evaluation
│   ├── observability/       # MLflow tracing configuration
│   ├── repositories/        # Database access and retrieval queries
│   ├── services/            # Pipeline, retrieval, generation, and cache logic
│   └── workers/             # Celery application and tasks
└── tests/                   # Unit and integration tests
```

## Design decisions

- **PostgreSQL for both metadata and retrieval:** relational data, full-text
  search, and vectors remain in one transactionally consistent system.
- **Hybrid retrieval before generation:** exact financial language and semantic
  similarity complement each other; RRF combines rankings without requiring
  directly comparable raw scores.
- **Cross-encoder only on candidates:** expensive pairwise scoring is restricted
  to a small retrieved set instead of the entire corpus.
- **Validated citations:** the generator can reference only chunks that were
  actually supplied as evidence.
- **Cache as an optimization:** Redis improves repeated-request latency but is
  not required for correctness.
- **Background work separated from web requests:** Celery owns long-running
  ingestion while FastAPI remains responsive.

## Current scope and roadmap

The complete Week 3 backend is implemented: ingestion, indexing, retrieval,
reranking, answer generation, citations, insufficient-evidence handling,
streaming, caching, background jobs, MLflow tracing, 50-question evaluation,
and adversarial-document testing.

The next milestones are:

- Streamlit research interface with source previews and query history
- Load testing, deployment, architecture assets, and a recorded product demo

## Evaluation limitations

The 50-question results currently cover one company and one 10-K filing.
They are useful for comparing retrieval configurations under controlled
conditions, but they do not yet establish performance across industries,
companies, filing formats, or time periods. Expanding this benchmark is an
explicit project milestone.

