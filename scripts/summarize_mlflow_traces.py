import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow

from finsight.core.config import get_settings

OUTPUT_PATH = Path("evals/results/mlflow_trace_summary.json")


def main() -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(
        settings.mlflow_experiment_name
    )
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment {settings.mlflow_experiment_name!r} was not found"
        )

    traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        max_results=10_000,
        return_type="list",
    )
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    input_cost = 0.0
    output_cost = 0.0
    total_cost = 0.0
    total_duration_ms = 0

    for trace in traces:
        metadata = trace.info.trace_metadata
        token_usage = json.loads(metadata.get("mlflow.trace.tokenUsage", "{}"))
        cost = json.loads(metadata.get("mlflow.trace.cost", "{}"))
        input_tokens += int(token_usage.get("input_tokens", 0))
        output_tokens += int(token_usage.get("output_tokens", 0))
        total_tokens += int(token_usage.get("total_tokens", 0))
        input_cost += float(cost.get("input_cost", 0.0))
        output_cost += float(cost.get("output_cost", 0.0))
        total_cost += float(cost.get("total_cost", 0.0))
        total_duration_ms += int(
            trace.info.to_dict().get("execution_duration_ms") or 0
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_name": settings.mlflow_experiment_name,
        "trace_count": len(traces),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
        "total_trace_duration_ms": total_duration_ms,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
