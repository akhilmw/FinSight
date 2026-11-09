import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI

from finsight.core.config import get_settings
from finsight.db.session import SessionLocal
from finsight.embeddings import OpenAIEmbeddingClient
from finsight.evaluation.answer import (
    OpenAIAnswerJudge,
    run_answer_evaluation,
    summarize_answer_results,
)
from finsight.evaluation.dataset import (
    load_retrieval_eval_cases,
    resolve_retrieval_eval_cases,
)
from finsight.generation import OpenAIAnswerGenerator
from finsight.observability import configure_mlflow_tracing
from finsight.repositories.filing_chunk import FilingChunkRepository
from finsight.reranking.cross_encoder import CrossEncoderReranker
from finsight.services.filing_answer import FilingAnswerService
from finsight.services.filing_search import FilingSearchService


class RepositoryEvidenceLoader:
    def __init__(self, repository: FilingChunkRepository) -> None:
        self._repository = repository

    def load(self, chunk_ids: list[int]) -> list[str]:
        return [chunk.text for chunk in self._repository.list_by_ids(chunk_ids)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FinSight answer quality and citations.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/retrieval_questions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results/answer_evaluation.json"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases for a smoke test.",
    )
    return parser.parse_args()


async def evaluate(
    dataset_path: Path,
    output_path: Path,
    limit: int | None,
) -> None:
    settings = get_settings()
    if settings.openai_api_key is None:
        raise RuntimeError("OpenAI is not configured")

    configure_mlflow_tracing(settings)
    cases = load_retrieval_eval_cases(dataset_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        cases = cases[:limit]

    async with AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
    ) as sdk_client:
        with SessionLocal() as session:
            cases = resolve_retrieval_eval_cases(cases, session)
            search_service = FilingSearchService(
                session=session,
                embedding_client=OpenAIEmbeddingClient(
                    client=sdk_client,
                    model=settings.embedding_model,
                    dimensions=settings.embedding_dimensions,
                ),
                reranker=CrossEncoderReranker(
                    model_name=settings.reranker_model,
                ),
            )
            answer_service = FilingAnswerService(
                search_service=search_service,
                answer_generator=OpenAIAnswerGenerator(
                    client=sdk_client,
                    model=settings.answer_model,
                ),
            )
            results = await run_answer_evaluation(
                cases=cases,
                answer_service=answer_service,
                judge=OpenAIAnswerJudge(
                    client=sdk_client,
                    model=settings.answer_model,
                ),
                evidence_loader=RepositoryEvidenceLoader(
                    FilingChunkRepository(session)
                ),
            )

    summary = summarize_answer_results(results)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "answer_model": settings.answer_model,
        "question_count": summary.question_count,
        "summary": {
            "sufficient_evidence_rate": summary.sufficient_evidence_rate,
            "mean_groundedness": summary.mean_groundedness,
            "mean_citation_correctness": summary.mean_citation_correctness,
            "mean_latency_ms": summary.mean_latency_ms,
        },
        "results": [
            {
                "evaluation_id": result.evaluation_id,
                "answer": result.answer,
                "sufficient_evidence": result.sufficient_evidence,
                "cited_chunk_ids": result.cited_chunk_ids,
                "groundedness": result.groundedness,
                "citation_correctness": result.citation_correctness,
                "latency_ms": result.latency_ms,
                "explanation": result.explanation,
            }
            for result in results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Questions:            {summary.question_count}")
    print(f"Sufficient evidence:  {summary.sufficient_evidence_rate:.3f}")
    print(f"Groundedness:         {summary.mean_groundedness:.3f}")
    print(f"Citation correctness: {summary.mean_citation_correctness:.3f}")
    print(f"Mean latency:         {summary.mean_latency_ms:.1f} ms")
    print(f"Results:              {output_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(evaluate(args.dataset, args.output, args.limit))


if __name__ == "__main__":
    main()
