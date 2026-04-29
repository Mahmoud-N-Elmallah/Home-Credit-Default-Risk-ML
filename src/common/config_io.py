from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import yaml

from src.common.env import load_project_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_hydra_config(overrides=None):
    load_project_dotenv()
    config_dir = PROJECT_ROOT / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides or [])
    return OmegaConf.to_container(cfg, resolve=True)


def save_yaml(path, payload, *, sort_keys=False):
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=sort_keys)
