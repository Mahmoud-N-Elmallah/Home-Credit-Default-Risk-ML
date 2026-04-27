import logging
import re
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import yaml

from src.common.artifacts import model_artifact_path, training_artifact_relative_path
from src.model_training.config import get_primary_estimator_config, primary_model_name


logger = logging.getLogger(__name__)

MAX_PARAM_LENGTH = 500
MAX_TAG_LENGTH = 5000


class NullTracker:
    def log_final(self, metadata):
        return


def mlflow_config(config):
    return config.get("tracking", {}).get("mlflow", {})


def mlflow_enabled(config):
    return bool(mlflow_config(config).get("enabled", False))


def truncate(value, limit):
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def safe_param(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return truncate(value, MAX_PARAM_LENGTH)
    return truncate(yaml.safe_dump(value, sort_keys=True), MAX_PARAM_LENGTH)


def dvc_remote_url():
    config_path = Path(".dvc") / "config"
    if not config_path.exists():
        return None
    match = re.search(r"^\s*url\s*=\s*(.+?)\s*$", config_path.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else None


def dagshub_repo_from_uri(tracking_uri):
    parsed = urlparse(tracking_uri)
    if parsed.netloc.lower() != "dagshub.com":
        return None
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2 or not parts[1].endswith(".mlflow"):
        return None
    return parts[0], parts[1].removesuffix(".mlflow")


def configure_tracking_backend(config_section):
    tracking_uri = config_section["tracking_uri"]
    repo = dagshub_repo_from_uri(tracking_uri)
    if repo:
        import dagshub

        dagshub.init(repo_owner=repo[0], repo_name=repo[1], mlflow=True)
        return

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)


def numeric_items(payload, prefix=""):
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from numeric_items(value, name)
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        yield prefix, float(payload)


def base_params(config):
    training = config["training"]
    params = {
        "primary_model": primary_model_name(config),
        "cv_splits": training["cv_splits"],
        "optuna_n_trials": training["optuna_n_trials"],
        "optuna_subsample_rate": training["optuna_subsample_rate"],
        "run_full_oof_validation": training["run_full_oof_validation"],
        "optimization_metric": training["optimization_metric"],
        "cv_shuffle": training["cv_shuffle"],
        "phases.search": training["phases"]["search"],
        "phases.validate": training["phases"]["validate"],
        "phases.final_fit": training["phases"]["final_fit"],
        "threshold_objective": training["threshold_tuning"]["objective"],
        "accelerator_preferred": training.get("accelerator", "gpu"),
        "accelerator_fallback": "cpu",
        "preprocessing.scaler": training["preprocessing"]["scaler"],
        "preprocessing.imbalance.strategy": training["preprocessing"]["imbalance"]["strategy"],
        "feature_selection.enabled_during_search": training["preprocessing"]["feature_selection"]["enabled_during_search"],
        "feature_selection.max_features": training["preprocessing"]["feature_selection"]["max_features"],
    }
    primary = get_primary_estimator_config(config)
    for key, value in primary.get("params", {}).items():
        params[f"model.{primary['name']}.{key}"] = value
    return params


def base_tags(config, metadata, models_dir):
    tags = {
        "experiment_id": metadata["experiment_id"],
        "primary_model": metadata["primary_model"],
        "artifact_dir": str(Path(models_dir)),
        "config_hash": metadata["config_hash"],
    }
    if metadata.get("data_hashes"):
        tags["train_data_hash"] = next(iter(metadata["data_hashes"].values()))
    remote_url = dvc_remote_url()
    if remote_url:
        tags["dvc_remote_url"] = remote_url
    return tags


class MlflowTracker:
    def __init__(self, mlflow_module, config, models_dir):
        self.mlflow = mlflow_module
        self.config = config
        self.models_dir = Path(models_dir)

    def log_final(self, metadata):
        self.mlflow.log_params({key: safe_param(value) for key, value in base_params(self.config).items()})
        self.mlflow.set_tags({key: truncate(value, MAX_TAG_LENGTH) for key, value in base_tags(self.config, metadata, self.models_dir).items()})
        self._log_metrics()
        if mlflow_config(self.config).get("log_artifacts", True):
            self._log_artifacts()

    def _log_metrics(self):
        metrics_path = model_artifact_path(self.models_dir, self.config, "metrics")
        if not metrics_path.exists():
            return
        metrics = yaml.safe_load(metrics_path.read_text(encoding="utf-8")) or {}
        flattened = dict(numeric_items(metrics))
        if flattened:
            self.mlflow.log_metrics(flattened)

    def _log_artifacts(self):
        keys = [
            "config_snapshot",
            "best_params",
            "threshold",
            "submission",
            "evaluation_report",
            "metrics",
            "threshold_table",
            "feature_importance",
            "feature_importance_plot",
            "confusion_matrix",
            "roc_curve",
        ]
        if mlflow_config(self.config).get("log_model_artifacts", True):
            keys.extend(["single_model", "preprocessor"])

        for key in keys:
            path = model_artifact_path(self.models_dir, self.config, key, model_name=primary_model_name(self.config))
            if path.exists() and path.is_file():
                artifact_path = None
                if path.parent != self.models_dir:
                    artifact_path = str(path.parent.relative_to(self.models_dir))
                self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

        metadata_path = self.models_dir / training_artifact_relative_path("run_metadata")
        if metadata_path.exists():
            self.mlflow.log_artifact(str(metadata_path))

        logs_dir = self.models_dir / "logs"
        if logs_dir.exists():
            self.mlflow.log_artifacts(str(logs_dir), artifact_path="logs")


@contextmanager
def tracking_run(config, models_dir, metadata):
    if not mlflow_enabled(config):
        yield NullTracker()
        return

    import mlflow

    config_section = mlflow_config(config)
    configure_tracking_backend(config_section)
    mlflow.set_experiment(config_section["experiment_name"])
    mlflow.start_run(run_name=metadata["experiment_id"])
    tracker = MlflowTracker(mlflow, config, models_dir)
    try:
        yield tracker
        mlflow.end_run(status="FINISHED")
    except Exception as error:
        logger.exception("MLflow-tracked training run failed.")
        mlflow.set_tag("error", truncate(error, MAX_TAG_LENGTH))
        mlflow.end_run(status="FAILED")
        raise
