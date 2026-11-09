import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI

from finsight.core.config import get_settings
from finsight.evaluation.security import (
    load_prompt_injection_cases,
    run_prompt_injection_evaluation,
)
from finsight.generation import OpenAIAnswerGenerator
from finsight.observability import configure_mlflow_tracing

DATASET_PATH = Path("evals/prompt_injection_cases.jsonl")
OUTPUT_PATH = Path("evals/results/prompt_injection_evaluation.json")


async def evaluate() -> None:
    settings = get_settings()
    if settings.openai_api_key is None:
        raise RuntimeError("OpenAI is not configured")

    configure_mlflow_tracing(settings)
    cases = load_prompt_injection_cases(DATASET_PATH)
    async with AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
    ) as sdk_client:
        results = await run_prompt_injection_evaluation(
            cases,
            OpenAIAnswerGenerator(sdk_client, settings.answer_model),
        )

    passed = sum(result.passed for result in results)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": settings.answer_model,
        "case_count": len(results),
        "passed": passed,
        "pass_rate": passed / len(results),
        "results": [
            {
                "evaluation_id": result.evaluation_id,
                "passed": result.passed,
                "answer": result.answer,
                "sufficient_evidence": result.sufficient_evidence,
                "triggered_phrases": result.triggered_phrases,
            }
            for result in results
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Prompt-injection pass rate: {passed}/{len(results)}")
    print(f"Results: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(evaluate())
