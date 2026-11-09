import asyncio
from datetime import date

from finsight.evaluation.answer import (
    StructuredAnswerGrade,
    run_answer_evaluation,
    summarize_answer_results,
)
from finsight.evaluation.dataset import RetrievalEvalCase
from finsight.services.filing_answer import AnswerCitation, FilingAnswerResult


class FakeAnswerService:
    async def answer(
        self,
        question: str,
        evidence_limit: int = 5,
        filing_id: int | None = None,
        company_name: str | None = None,
        year: int | None = None,
        form_type: str | None = None,
    ) -> FilingAnswerResult:
        return FilingAnswerResult(
            question=question,
            answer="Revenue was $10 billion.",
            sufficient_evidence=True,
            citations=[
                AnswerCitation(
                    chunk_id=7,
                    filing_id=3,
                    accession_number="accession",
                    company_name="NVIDIA",
                    form_type="10-K",
                    filing_date=date(2026, 2, 25),
                    source_url="https://www.sec.gov/example.htm",
                    item="7",
                    title="MD&A",
                )
            ],
        )


class FakeJudge:
    async def grade(
        self,
        question: str,
        reference_answer: str,
        answer: str,
        cited_evidence: list[str],
    ) -> StructuredAnswerGrade:
        assert cited_evidence == ["Revenue was $10 billion."]
        return StructuredAnswerGrade(
            groundedness=1.0,
            citation_correctness=0.9,
            explanation="The citation supports the answer.",
        )


class FakeEvidenceLoader:
    def load(self, chunk_ids: list[int]) -> list[str]:
        assert chunk_ids == [7]
        return ["Revenue was $10 billion."]


def create_case() -> RetrievalEvalCase:
    return RetrievalEvalCase.model_validate(
        {
            "id": "fin-001",
            "question": "What was revenue?",
            "query_type": "exact",
            "filters": {"company_name": "NVIDIA", "year": 2026},
            "relevant_chunks": [
                {
                    "accession_number": "accession",
                    "item": "7",
                    "section_title": "MD&A",
                    "chunk_index": 0,
                    "current_chunk_id": 7,
                }
            ],
            "answer_hint": "Revenue was $10 billion.",
        }
    )


def test_runs_and_summarizes_answer_evaluation() -> None:
    results = asyncio.run(
        run_answer_evaluation(
            cases=[create_case()],
            answer_service=FakeAnswerService(),
            judge=FakeJudge(),
            evidence_loader=FakeEvidenceLoader(),
        )
    )
    summary = summarize_answer_results(results)

    assert results[0].cited_chunk_ids == [7]
    assert summary.question_count == 1
    assert summary.sufficient_evidence_rate == 1.0
    assert summary.mean_groundedness == 1.0
    assert summary.mean_citation_correctness == 0.9
