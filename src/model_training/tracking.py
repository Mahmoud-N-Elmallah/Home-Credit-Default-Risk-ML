import logging
import re
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import joblib
import mlflow.pyfunc
import pandas as pd
import yaml

from src.common.artifacts import model_artifact_path, training_artifact_relative_path
from src.common.schema import clean_column_names, expected_preprocessor_input_columns
from src.model_training.config import get_primary_estimator_config, primary_model_name


logger = logging.getLogger(__name__)

MAX_PARAM_LENGTH = 500
MAX_TAG_LENGTH = 5000
PYFUNC_ARTIFACT_PATH = "pyfunc_model"


class NullTracker:
    def log_final(self, metadata):
        return

    def log_start(self, metadata):
        return


class CreditRiskPyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, preprocessor, id_col, target_col, probability_col):
        self.model = model
        self.preprocessor = preprocessor
        self.id_col = id_col
        self.target_col = target_col
        self.probability_col = probability_col

    def predict(self, context, model_input):
        frame = pd.DataFrame(model_input).copy()
        drop_cols = [col for col in [self.id_col, self.target_col] if col in frame.columns]
        features = clean_column_names(frame.drop(columns=drop_cols))
        expected_columns = expected_preprocessor_input_columns(self.preprocessor)
        if expected_columns is not None:
            features = features.reindex(columns=expected_columns, fill_value=0)
        processed = self.preprocessor.transform(features)
        probabilities = self.model.predict_proba(processed)[:, 1]
        return pd.DataFrame({self.probability_col: probabilities})


def mlflow_config(config):
    return config.get("tracking", {}).get("mlflow", {})


def mlflow_enabled(config):
    return bool(mlflow_config(config).get("enabled", False))


def registry_config(config):
    return mlflow_config(config).get("registry", {})


def registry_enabled(config):
    return bool(registry_config(config).get("enabled", False))


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


def load_metrics(models_dir, config):
    metrics_path = model_artifact_path(models_dir, config, "metrics")
    if not metrics_path.exists():
        return {}
    return yaml.safe_load(metrics_path.read_text(encoding="utf-8")) or {}


def ranking_roc_auc(metrics):
    try:
        return float(metrics["ranking"]["roc_auc"])
    except (KeyError, TypeError, ValueError):
        return None


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

    def log_start(self, metadata):
        self.mlflow.log_params({key: safe_param(value) for key, value in base_params(self.config).items()})
        self.mlflow.set_tags({key: truncate(value, MAX_TAG_LENGTH) for key, value in base_tags(self.config, metadata, self.models_dir).items()})

    def log_final(self, metadata):
        metrics = self._log_metrics()
        if mlflow_config(self.config).get("log_artifacts", True):
            self._log_artifacts()
        if registry_enabled(self.config):
            self._log_registered_model(metadata, metrics)

    def _log_metrics(self):
        metrics = load_metrics(self.models_dir, self.config)
        flattened = dict(numeric_items(metrics))
        if flattened:
            self.mlflow.log_metrics(flattened)
        return metrics

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

    def _log_registered_model(self, metadata, metrics):
        registry = registry_config(self.config)
        roc_auc = ranking_roc_auc(metrics)
        min_roc_auc = float(registry.get("min_roc_auc", 0.0))
        if roc_auc is None:
            self.mlflow.set_tag("registry_status", "skipped_missing_roc_auc")
            return
        if roc_auc < min_roc_auc:
            self.mlflow.set_tag("registry_status", "skipped_metric_gate")
            self.mlflow.set_tag("registry_min_roc_auc", str(min_roc_auc))
            return

        try:
            model_info = self._log_pyfunc_model(registry)
            self._tag_registered_model(registry, model_info, metadata)
            self.mlflow.set_tag("registry_status", "registered")
        except Exception as error:
            self.mlflow.set_tag("registry_status", "failed")
            self.mlflow.set_tag("registry_error", truncate(error, MAX_TAG_LENGTH))
            if registry.get("required", False):
                raise
            logger.warning("MLflow model registry step failed: %s", error)

    def _log_pyfunc_model(self, registry):
        from mlflow.models import infer_signature

        model_path = model_artifact_path(self.models_dir, self.config, "single_model", model_name=primary_model_name(self.config))
        preprocessor_path = model_artifact_path(self.models_dir, self.config, "preprocessor")
        if not model_path.exists() or not preprocessor_path.exists():
            raise FileNotFoundError("Final model or preprocessor artifact is missing; cannot log registry model.")

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        input_example = self._registry_input_example()
        pyfunc_model = CreditRiskPyFuncModel(
            model,
            preprocessor,
            self.config["training"]["id_col"],
            self.config["training"]["target_col"],
            self.config["inference"]["probability_col"],
        )
        prediction_example = pyfunc_model.predict(None, input_example)
        signature = infer_signature(input_example, prediction_example)
        return self.mlflow.pyfunc.log_model(
            artifact_path=PYFUNC_ARTIFACT_PATH,
            python_model=pyfunc_model,
            input_example=input_example,
            signature=signature,
            registered_model_name=registry["registered_model_name"],
        )

    def _registry_input_example(self):
        train_path = Path(self.config["data"]["final"]["train"])
        sample = pd.read_csv(train_path, nrows=20)
        target_col = self.config["training"]["target_col"]
        if target_col in sample.columns:
            sample = sample.drop(columns=[target_col])
        return sample

    def _tag_registered_model(self, registry, model_info, metadata):
        version = getattr(model_info, "registered_model_version", None)
        if version is None:
            return
        client = self.mlflow.tracking.MlflowClient()
        model_name = registry["registered_model_name"]
        tags = base_tags(self.config, metadata, self.models_dir)
        for key, value in tags.items():
            try:
                client.set_model_version_tag(model_name, version, key, truncate(value, MAX_TAG_LENGTH))
            except Exception as error:
                logger.warning("Could not set MLflow model version tag %s=%s: %s", key, value, error)
        alias = registry.get("alias")
        if alias:
            try:
                client.set_registered_model_alias(model_name, alias, version)
            except Exception as error:
                logger.warning("Could not set MLflow registered model alias %s -> %s: %s", alias, version, error)


def end_mlflow_run(mlflow_module, status):
    try:
        mlflow_module.end_run(status=status)
    except UnicodeEncodeError as error:
        logger.warning("MLflow ended the run but failed to print its run URL on this terminal encoding: %s", error)


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
    tracker.log_start(metadata)
    try:
        yield tracker
        end_mlflow_run(mlflow, status="FINISHED")
    except Exception as error:
        logger.exception("MLflow-tracked training run failed.")
        mlflow.set_tag("error", truncate(error, MAX_TAG_LENGTH))
        end_mlflow_run(mlflow, status="FAILED")
        raise
