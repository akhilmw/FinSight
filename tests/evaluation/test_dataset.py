import json
from pathlib import Path

import pytest

from finsight.evaluation.dataset import load_retrieval_eval_cases


def create_case(
    case_id: str = "fin-001",
) -> dict[str, object]:
    return {
        "id": case_id,
        "question": "What are NVIDIA's operating segments?",
        "query_type": "exact",
        "filters": {
            "company_name": "NVIDIA",
            "year": 2026,
            "form_type": "10-K",
        },
        "relevant_chunks": [
            {
                "accession_number": "0001045810-26-000021",
                "item": "1",
                "section_title": "Business",
                "chunk_index": 3,
                "current_chunk_id": 490,
            }
        ],
        "answer_hint": "Compute & Networking and Graphics.",
    }


def write_jsonl(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
    )


def test_loads_valid_cases_and_ignores_blank_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evaluation.jsonl"
    path.write_text(
        f"{json.dumps(create_case('fin-001'))}\n\n"
        f"{json.dumps(create_case('fin-002'))}\n"
    )

    cases = load_retrieval_eval_cases(path)

    assert [case.id for case in cases] == ["fin-001", "fin-002"]
    assert cases[0].filters.company_name == "NVIDIA"
    assert cases[0].relevant_chunk_ids == {490}


def test_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.jsonl"
    write_jsonl(
        path,
        [create_case("fin-001"), create_case("fin-001")],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate evaluation case ID: fin-001",
    ):
        load_retrieval_eval_cases(path)


@pytest.mark.parametrize(
    "content",
    [
        "not valid JSON",
        json.dumps({**create_case(), "id": "question-1"}),
        json.dumps({**create_case(), "query_type": "unknown"}),
        json.dumps({**create_case(), "relevant_chunks": []}),
    ],
)
def test_rejects_invalid_case_data(
    tmp_path: Path,
    content: str,
) -> None:
    path = tmp_path / "evaluation.jsonl"
    path.write_text(content)

    with pytest.raises(
        ValueError,
        match="Invalid evaluation case on line 1",
    ):
        load_retrieval_eval_cases(path)


@pytest.mark.parametrize("content", ["", "\n\n"])
def test_rejects_empty_dataset(
    tmp_path: Path,
    content: str,
) -> None:
    path = tmp_path / "evaluation.jsonl"
    path.write_text(content)

    with pytest.raises(
        ValueError,
        match="Evaluation dataset must not be empty",
    ):
        load_retrieval_eval_cases(path)
