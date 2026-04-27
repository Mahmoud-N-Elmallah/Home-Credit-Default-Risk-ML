from datetime import datetime
from hashlib import sha256
from pathlib import Path
import re
import shutil

import yaml

from src.common.artifacts import model_artifact_path
from src.model_training.config import primary_model_name


def stable_yaml_hash(data):
    dumped = yaml.safe_dump(data, sort_keys=True)
    return sha256(dumped.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return slug.strip("_") or "experiment"


def create_experiment_dir(models_root, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_config = config["training"]
    exp_config = t_config["experiment"]
    if exp_config.get("name"):
        experiment_id = slugify(exp_config["name"])
        experiment_dir = models_root / experiment_id
        if exp_config.get("overwrite_existing", False) and experiment_dir.exists():
            shutil.rmtree(experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir, experiment_dir.name, timestamp
    else:
        experiment_id = exp_config["folder_template"].format(
            timestamp=timestamp,
            primary_model=primary_model_name(config),
        )
        experiment_id = slugify(experiment_id)

        experiment_dir = models_root / experiment_id
        suffix = 1
        while experiment_dir.exists():
            suffix += 1
            experiment_dir = models_root / f"{experiment_id}_{suffix}"
        experiment_dir.mkdir(parents=True, exist_ok=False)
        return experiment_dir, experiment_dir.name, timestamp


def write_latest_experiment_pointer(models_root, config, experiment_dir):
    latest_path = Path(config["training"]["artifact_paths"]["latest_experiment"])
    if not latest_path.is_absolute():
        latest_path = models_root / latest_path
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(experiment_dir.resolve()), encoding="utf-8")


def save_config_snapshot(experiment_dir, config):
    path = model_artifact_path(experiment_dir, config, "config_snapshot")
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def metadata_path(models_dir, config):
    metadata_name = config["training"]["artifact_reuse"]["metadata"]
    path = Path(metadata_name)
    return path if path.is_absolute() else models_dir / path


def build_run_metadata(config, X, y, train_path, experiment_id, timestamp):
    phases = config["training"]["phases"]
    metric_scopes = []
    if phases["search"]:
        metric_scopes.append("search_subsample_cv")
    if phases["validate"] and config["training"]["run_full_oof_validation"]:
        metric_scopes.append("out_of_fold")
    if phases["final_fit"]:
        metric_scopes.append("final_train_fit")

    return {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "config_hash": stable_yaml_hash(config),
        "data_hashes": {str(train_path): file_hash(train_path)},
        "primary_model": primary_model_name(config),
        "phases": phases,
        "cv_splits": config["training"]["cv_splits"],
        "optuna_n_trials": config["training"]["optuna_n_trials"],
        "optuna_subsample_rate": config["training"]["optuna_subsample_rate"],
        "row_count": int(len(X)),
        "feature_count": int(X.shape[1]),
        "positive_count": int(y.sum()),
        "metric_scopes": metric_scopes,
        "selected_accelerators": {},
        "chosen_threshold": None,
        "artifact_list": [],
    }


def write_run_metadata(models_dir, config, metadata):
    metadata_file_name = config["training"]["artifact_reuse"]["metadata"]
    artifact_list = [
        str(path.relative_to(models_dir)).replace("\\", "/")
        for path in models_dir.rglob("*")
        if path.is_file() and path.name != metadata_file_name
    ]
    path = metadata_path(models_dir, config)
    artifact_list.append(str(path.relative_to(models_dir)).replace("\\", "/"))
    metadata["artifact_list"] = sorted(artifact_list)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)


def validate_reusable_artifacts(models_dir, config, metadata):
    if config["training"]["artifact_reuse"]["allow_stale_artifacts"]:
        return
    path = metadata_path(models_dir, config)
    if not path.exists():
        return
    existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if existing.get("config_hash") != metadata["config_hash"]:
        raise ValueError("Existing training artifacts use different config. Set allow_stale_artifacts=true to reuse.")
    if existing.get("data_hashes") != metadata["data_hashes"]:
        raise ValueError("Existing training artifacts use different data. Set allow_stale_artifacts=true to reuse.")
