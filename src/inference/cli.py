import argparse

from src.common.config_io import load_hydra_config
from src.inference.core import run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the saved best Home Credit model.")
    parser.add_argument("--experiment-dir", help="Experiment directory. Defaults to artifacts.best_experiment_dir.")
    parser.add_argument("--input", help="CSV input with processed pipeline features.")
    parser.add_argument("--json", help="JSON object or list of objects with processed pipeline features.")
    parser.add_argument(
        "--output",
        help="Output CSV path. Relative paths are resolved inside the experiment dir.",
    )
    parser.add_argument("--threshold", type=float, help="Override classification threshold for binary labels.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example inference.input_path=data.csv.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_hydra_config(args.overrides)
    run_inference(config, args.experiment_dir, args.input, args.json, args.output, args.threshold)
