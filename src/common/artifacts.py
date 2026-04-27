from pathlib import Path


DEFAULT_MODELS_DIR = "Models"

TRAINING_ARTIFACT_PATHS = {
    "latest_experiment": "latest_experiment.txt",
    "config_snapshot": "config_snapshot.yaml",
    "submission": "submission.csv",
    "preprocessor": "training_preprocessor.pkl",
    "threshold": "threshold.yaml",
    "best_params": "best_params.yaml",
    "single_model": "final_model.pkl",
    "evaluation_report": "reports/evaluation_report.txt",
    "metrics": "reports/metrics.yaml",
    "threshold_table": "reports/threshold_table.csv",
    "feature_importance": "reports/feature_importance.csv",
    "feature_importance_plot": "plots/feature_importance_top.png",
    "confusion_matrix": "plots/confusion_matrix.png",
    "roc_curve": "plots/roc_curve.png",
    "run_metadata": "training_run_metadata.yaml",
}


def training_models_dir(config):
    return Path(config.get("training", {}).get("artifact_paths", {}).get("models_dir", DEFAULT_MODELS_DIR))


def training_artifact_template(key):
    try:
        return TRAINING_ARTIFACT_PATHS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown training artifact key: {key}") from exc


def training_artifact_relative_path(key, **kwargs):
    return Path(training_artifact_template(key).format(**kwargs))


def model_artifact_path(models_dir, config, key, **kwargs):
    path = training_artifact_relative_path(key, **kwargs)
    resolved = path if path.is_absolute() else Path(models_dir) / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
