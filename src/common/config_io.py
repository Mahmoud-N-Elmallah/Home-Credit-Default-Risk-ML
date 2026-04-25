from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(path, payload, *, sort_keys=False):
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=sort_keys)
