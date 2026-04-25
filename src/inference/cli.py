import argparse

from src.common.config_io import resolve_project_path as resolve_path
from src.inference.core import run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the saved best Home Credit model.")
    parser.add_argument("--config", default="config.yaml", help="Path to project config YAML.")
    parser.add_argument("--experiment-dir", help="Experiment directory. Defaults to artifacts.best_experiment_dir.")
    parser.add_argument("--input", help="CSV input with processed pipeline features.")
    parser.add_argument("--json", help="JSON object or list of objects with processed pipeline features.")
    parser.add_argument(
        "--output",
        help="Output CSV path. Relative paths are resolved inside the experiment dir.",
    )
    parser.add_argument("--threshold", type=float, help="Override classification threshold for binary labels.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    run_inference(config_path, args.experiment_dir, args.input, args.json, args.output, args.threshold)
