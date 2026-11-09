import mlflow
from mlflow import openai as mlflow_openai

from finsight.core.config import Settings


def configure_mlflow_tracing(settings: Settings) -> bool:
    """Configure OpenAI tracing when explicitly enabled.

    Returning whether tracing was enabled makes startup behavior easy to test
    without requiring a running MLflow server.
    """
    if not settings.mlflow_tracing_enabled:
        return False

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    mlflow_openai.autolog(silent=True)
    return True
