from unittest.mock import MagicMock

from finsight.observability import tracing


def create_settings(enabled: bool) -> MagicMock:
    settings = MagicMock()
    settings.mlflow_tracing_enabled = enabled
    settings.mlflow_tracking_uri = "http://mlflow:5000"
    settings.mlflow_experiment_name = "finsight-test"
    return settings


def test_tracing_remains_disabled_by_default(
    monkeypatch,
) -> None:
    set_tracking_uri = MagicMock()
    monkeypatch.setattr(tracing.mlflow, "set_tracking_uri", set_tracking_uri)

    enabled = tracing.configure_mlflow_tracing(create_settings(False))

    assert enabled is False
    set_tracking_uri.assert_not_called()


def test_tracing_configures_mlflow_and_openai(
    monkeypatch,
) -> None:
    set_tracking_uri = MagicMock()
    set_experiment = MagicMock()
    autolog = MagicMock()
    monkeypatch.setattr(tracing.mlflow, "set_tracking_uri", set_tracking_uri)
    monkeypatch.setattr(tracing.mlflow, "set_experiment", set_experiment)
    monkeypatch.setattr(tracing.mlflow_openai, "autolog", autolog)

    enabled = tracing.configure_mlflow_tracing(create_settings(True))

    assert enabled is True
    set_tracking_uri.assert_called_once_with("http://mlflow:5000")
    set_experiment.assert_called_once_with("finsight-test")
    autolog.assert_called_once_with(silent=True)
