from pathlib import Path
import logging
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Ensure src module can be found
sys.path.append(str(Path(__file__).parent))
from src.common.logging import configure_logging
from src.data_processing.run_pipeline import run_pipeline
from src.model_training.run_training import run_training


logger = logging.getLogger(__name__)


def configure_hydra_run_logging():
    hydra_dir = HydraConfig.get().runtime.output_dir
    return configure_logging(Path(hydra_dir) / "logs", "pipeline.log")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    log_path = configure_hydra_run_logging()
    config = OmegaConf.to_container(cfg, resolve=True)
    step = config.get("run", {}).get("step", "all")

    if step not in {"process", "train", "all"}:
        raise ValueError(f"Invalid run.step: {step}. Expected one of: process, train, all.")

    if step in {"process", "all"}:
        logger.info("Starting Data Processing Pipeline")
        run_pipeline(config)

    if step in {"train", "all"}:
        logger.info("Starting Model Training Pipeline")
        run_training(config)

    logger.info("Pipeline finished. Log: %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
