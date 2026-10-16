from dataclasses import dataclass
from statistics import fmean

from finsight.evaluation.runner import (
    RETRIEVAL_MODES,
    RetrievalEvalResult,
)
from finsight.schemas import SearchMode


@dataclass(frozen=True)
class RetrievalModeSummary:
    mode: SearchMode
    question_count: int
    mean_recall_at_5: float
    mean_recall_at_10: float
    mean_reciprocal_rank: float
    mean_latency_ms: float


def summarize_retrieval_results(
    results: list[RetrievalEvalResult],
) -> list[RetrievalModeSummary]:
    if not results:
        raise ValueError("results must not be empty")

    summaries: list[RetrievalModeSummary] = []

    for mode in RETRIEVAL_MODES:
        mode_results = [
            result
            for result in results
            if result.mode is mode
        ]

        if not mode_results:
            continue

        summaries.append(
            RetrievalModeSummary(
                mode=mode,
                question_count=len(mode_results),
                mean_recall_at_5=fmean(
                    result.recall_at_5
                    for result in mode_results
                ),
                mean_recall_at_10=fmean(
                    result.recall_at_10
                    for result in mode_results
                ),
                mean_reciprocal_rank=fmean(
                    result.reciprocal_rank
                    for result in mode_results
                ),
                mean_latency_ms=fmean(
                    result.latency_ms
                    for result in mode_results
                ),
            )
        )

    return summaries