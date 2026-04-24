from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure src module can be found
sys.path.append(str(Path(__file__).parent))
from src.data_processing.run_pipeline import run_pipeline
from src.model_training.run_training import run_training


LEGACY_STEP_FLAGS = {
    "--process": "process",
    "--train": "train",
    "--all": "all",
}


def _config_path_args(config_file):
    path = Path(config_file)
    config_dir = path.parent if str(path.parent) else Path(".")
    config_name = path.name
    if path.suffix in [".yaml", ".yml"]:
        config_name = path.stem
    return ["--config-path", str(config_dir), "--config-name", config_name]


def rewrite_legacy_args(argv):
    """Translate old argparse flags to Hydra overrides before Hydra parses argv."""
    rewritten = [argv[0]]
    step_flags = []
    index = 1

    while index < len(argv):
        arg = argv[index]

        if arg in LEGACY_STEP_FLAGS:
            step_flags.append(LEGACY_STEP_FLAGS[arg])
            index += 1
            continue

        if arg == "--config":
            if index + 1 >= len(argv):
                raise ValueError("--config requires a config file path.")
            rewritten.extend(_config_path_args(argv[index + 1]))
            index += 2
            continue

        if arg.startswith("--config="):
            rewritten.extend(_config_path_args(arg.split("=", 1)[1]))
            index += 1
            continue

        rewritten.append(arg)
        index += 1

    if step_flags:
        step = "all" if "all" in step_flags or set(step_flags) == {"process", "train"} else step_flags[-1]
        rewritten.append(f"run.step={step}")

    return rewritten


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    step = config.get("run", {}).get("step", "all")

    if step not in {"process", "train", "all"}:
        raise ValueError(f"Invalid run.step: {step}. Expected one of: process, train, all.")

    if step in {"process", "all"}:
        print("\n--- Starting Data Processing Pipeline ---")
        run_pipeline(config)

    if step in {"train", "all"}:
        print("\n--- Starting Model Training Pipeline ---")
        run_training(config)

    print("\nPipeline finished.")
    return 0


if __name__ == "__main__":
    sys.argv = rewrite_legacy_args(sys.argv)
    sys.exit(main())
