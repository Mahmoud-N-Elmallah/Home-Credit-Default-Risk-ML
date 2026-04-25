from pathlib import Path


def model_artifact_path(models_dir, config, key, **kwargs):
    pattern = config["training"]["artifact_paths"][key]
    path = Path(pattern.format(**kwargs))
    resolved = path if path.is_absolute() else Path(models_dir) / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
