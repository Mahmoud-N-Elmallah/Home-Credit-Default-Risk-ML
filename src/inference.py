import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import yaml

from src.model_training.run_training import clean_column_names, model_artifact_path


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


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def experiment_path(config, experiment_dir_arg):
    return resolve_path(experiment_dir_arg or config["artifacts"]["best_experiment_dir"])


def output_path(config, experiment_dir, output_arg):
    configured = output_arg or config["inference"]["output_path"]
    path = Path(configured)
    return path if path.is_absolute() else experiment_dir / path


def load_input_frame(config, input_arg, json_arg):
    if input_arg and json_arg:
        raise ValueError("Use either --input or --json, not both.")

    input_path = input_arg or config["inference"].get("input_path")
    if json_arg:
        payload = json.loads(json_arg)
        return pd.DataFrame(payload if isinstance(payload, list) else [payload])
    if input_path:
        return pd.read_csv(resolve_path(input_path))
    raise ValueError("No inference input provided. Use --input, --json, or inference.input_path in config.")


def expected_preprocessor_input_columns(preprocessor):
    if getattr(preprocessor, "scaler", None) is not None and hasattr(preprocessor.scaler, "feature_names_in_"):
        return preprocessor.scaler.feature_names_in_.tolist()
    if getattr(preprocessor, "selector", None) is not None and hasattr(preprocessor.selector, "feature_names_in_"):
        return preprocessor.selector.feature_names_in_.tolist()
    return None


def prepare_features(df, preprocessor, config):
    id_col = config["training"]["id_col"]
    target_col = config["training"]["target_col"]
    inference_config = config["inference"]

    if target_col in df.columns and not inference_config["allow_target_column"]:
        raise ValueError(f"{target_col} found in inference input. Set inference.allow_target_column=true to drop it.")

    ids = df[id_col].copy() if id_col in df.columns else pd.Series(range(len(df)), name=id_col)
    drop_cols = [col for col in [id_col, target_col] if col in df.columns]
    X = clean_column_names(df.drop(columns=drop_cols))

    expected_columns = expected_preprocessor_input_columns(preprocessor)
    if expected_columns is None:
        return ids, X, {"missing_columns": [], "extra_columns": []}

    current_columns = set(X.columns)
    expected_set = set(expected_columns)
    missing_columns = [col for col in expected_columns if col not in current_columns]
    extra_columns = [col for col in X.columns if col not in expected_set]

    strategy = inference_config["missing_feature_strategy"]
    if missing_columns and strategy == "error":
        preview = missing_columns[:10]
        raise ValueError(f"Inference input is missing {len(missing_columns)} required features: {preview}")
    if strategy != "error" and strategy != "fill":
        raise ValueError(f"Unknown inference.missing_feature_strategy: {strategy}")
    max_missing = inference_config.get("max_missing_features_to_fill")
    if (
        missing_columns
        and strategy == "fill"
        and max_missing is not None
        and len(missing_columns) > int(max_missing)
    ):
        preview = missing_columns[:10]
        raise ValueError(
            f"Inference input is missing {len(missing_columns)} features, above "
            f"inference.max_missing_features_to_fill={max_missing}: {preview}"
        )

    fill_value = inference_config["missing_feature_fill_value"]
    X = X.reindex(columns=expected_columns, fill_value=fill_value)
    return ids, X, {"missing_columns": missing_columns, "extra_columns": extra_columns}


def load_threshold(config, experiment_dir, threshold_arg):
    if threshold_arg is not None:
        return threshold_arg

    inference_config = config["inference"]
    source = inference_config["threshold_source"]
    if source == "config":
        return float(inference_config["default_threshold"])
    if source != "artifact":
        raise ValueError(f"Unknown inference.threshold_source: {source}")

    threshold_path = model_artifact_path(experiment_dir, config, "threshold")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold artifact not found: {threshold_path}")
    threshold_info = load_yaml(threshold_path)
    return float(threshold_info["threshold"])


def run_inference(
    config_path,
    experiment_dir_arg=None,
    input_arg=None,
    json_arg=None,
    output_arg=None,
    threshold_arg=None,
):
    config = load_yaml(config_path)
    experiment_dir = experiment_path(config, experiment_dir_arg)
    artifact_paths = config["training"]["artifact_paths"]

    model_path = experiment_dir / artifact_paths["single_model"]
    preprocessor_path = experiment_dir / artifact_paths["preprocessor"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    if not hasattr(preprocessor, "pruned_columns"):
        preprocessor.pruned_columns = None
    df = load_input_frame(config, input_arg, json_arg)
    ids, X, alignment = prepare_features(df, preprocessor, config)
    X_processed = preprocessor.transform(X)
    probabilities = model.predict_proba(X_processed)[:, 1]

    inference_config = config["inference"]
    output = pd.DataFrame(
        {
            config["training"]["id_col"]: ids.to_numpy(),
            inference_config["probability_col"]: probabilities,
        }
    )
    if inference_config["include_binary_label"]:
        threshold = load_threshold(config, experiment_dir, threshold_arg)
        output[inference_config["label_col"]] = (output[inference_config["probability_col"]] >= threshold).astype(int)

    path = output_path(config, experiment_dir, output_arg)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)

    print(f"Inference saved to {path}")
    print(f"Rows scored: {len(output)}")
    if alignment["missing_columns"]:
        print(f"Missing features filled: {len(alignment['missing_columns'])}")
    if alignment["extra_columns"]:
        print(f"Extra input columns ignored: {len(alignment['extra_columns'])}")
    return output


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    run_inference(config_path, args.experiment_dir, args.input, args.json, args.output, args.threshold)


if __name__ == "__main__":
    main()
