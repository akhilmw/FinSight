import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI

from finsight.core.config import get_settings
from finsight.db.session import SessionLocal
from finsight.embeddings import OpenAIEmbeddingClient
from finsight.evaluation.dataset import load_retrieval_eval_cases
from finsight.evaluation.report import (
    RetrievalModeSummary,
    summarize_retrieval_results,
)
from finsight.evaluation.runner import (
    RetrievalEvalResult,
    run_retrieval_evaluation,
)
from finsight.services.filing_search import FilingSearchService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FinSight retrieval modes.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/retrieval_questions.jsonl"),
        help="Path to the retrieval evaluation JSONL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evals/results"),
        help="Directory where evaluation results will be written.",
    )
    return parser.parse_args()


def serialize_result(result: RetrievalEvalResult) -> dict[str, object]:
    return {
        "evaluation_id": result.evaluation_id,
        "query_type": result.query_type,
        "mode": result.mode.value,
        "retrieved_chunk_ids": result.retrieved_chunk_ids,
        "relevant_chunk_ids": sorted(result.relevant_chunk_ids),
        "recall_at_5": result.recall_at_5,
        "recall_at_10": result.recall_at_10,
        "reciprocal_rank": result.reciprocal_rank,
        "latency_ms": result.latency_ms,
    }


def serialize_summary(
    summary: RetrievalModeSummary,
) -> dict[str, object]:
    return {
        "mode": summary.mode.value,
        "question_count": summary.question_count,
        "mean_recall_at_5": summary.mean_recall_at_5,
        "mean_recall_at_10": summary.mean_recall_at_10,
        "mean_reciprocal_rank": summary.mean_reciprocal_rank,
        "mean_latency_ms": summary.mean_latency_ms,
    }


def write_results(
    output_directory: Path,
    generated_at: str,
    embedding_model: str,
    question_count: int,
    results: list[RetrievalEvalResult],
    summaries: list[RetrievalModeSummary],
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)

    detailed_path = output_directory / "retrieval_detailed.json"
    summary_path = output_directory / "retrieval_summary.json"

    metadata = {
        "generated_at": generated_at,
        "embedding_model": embedding_model,
        "question_count": question_count,
    }

    detailed_path.write_text(
        json.dumps(
            {
                **metadata,
                "result_count": len(results),
                "results": [
                    serialize_result(result)
                    for result in results
                ],
            },
            indent=2,
        )
        + "\n"
    )
    summary_path.write_text(
        json.dumps(
            {
                **metadata,
                "summaries": [
                    serialize_summary(summary)
                    for summary in summaries
                ],
            },
            indent=2,
        )
        + "\n"
    )

    return detailed_path, summary_path


def print_summary(summaries: list[RetrievalModeSummary]) -> None:
    print()
    print(
        f"{'Mode':<10} "
        f"{'Questions':>9} "
        f"{'Recall@5':>10} "
        f"{'Recall@10':>11} "
        f"{'MRR':>8} "
        f"{'Latency ms':>12}"
    )
    print("-" * 65)

    for summary in summaries:
        print(
            f"{summary.mode.value:<10} "
            f"{summary.question_count:>9} "
            f"{summary.mean_recall_at_5:>10.3f} "
            f"{summary.mean_recall_at_10:>11.3f} "
            f"{summary.mean_reciprocal_rank:>8.3f} "
            f"{summary.mean_latency_ms:>12.1f}"
        )


async def evaluate(
    dataset_path: Path,
    output_directory: Path,
) -> None:
    settings = get_settings()

    if settings.openai_api_key is None:
        raise RuntimeError("The embedding service is not configured")

    cases = load_retrieval_eval_cases(dataset_path)

    async with AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
    ) as sdk_client:
        embedding_client = OpenAIEmbeddingClient(
            client=sdk_client,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

        with SessionLocal() as session:
            search_service = FilingSearchService(
                session=session,
                embedding_client=embedding_client,
            )
            results = await run_retrieval_evaluation(
                cases=cases,
                search_service=search_service,
            )

    summaries = summarize_retrieval_results(results)
    generated_at = datetime.now(UTC).isoformat()
    detailed_path, summary_path = write_results(
        output_directory=output_directory,
        generated_at=generated_at,
        embedding_model=settings.embedding_model,
        question_count=len(cases),
        results=results,
        summaries=summaries,
    )

    print_summary(summaries)
    print()
    print(f"Detailed results: {detailed_path}")
    print(f"Summary results:  {summary_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(
        evaluate(
            dataset_path=args.dataset,
            output_directory=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
