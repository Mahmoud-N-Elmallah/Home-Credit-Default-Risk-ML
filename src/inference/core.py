import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.common.artifacts import model_artifact_path, training_artifact_relative_path
from src.common.config_io import resolve_project_path
from src.common.logging import configure_logging
from src.common.schema import clean_column_names, expected_preprocessor_input_columns

resolve_path = resolve_project_path
logger = logging.getLogger(__name__)


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
    config,
    experiment_dir_arg=None,
    input_arg=None,
    json_arg=None,
    output_arg=None,
    threshold_arg=None,
):
    experiment_dir = experiment_path(config, experiment_dir_arg)
    configure_logging(experiment_dir / "logs", "inference.log")

    model_path = experiment_dir / training_artifact_relative_path("single_model")
    preprocessor_path = experiment_dir / training_artifact_relative_path("preprocessor")
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

    logger.info("Inference saved to %s", path)
    logger.info("Rows scored: %s", len(output))
    if alignment["missing_columns"]:
        logger.info("Missing features filled: %s", len(alignment["missing_columns"]))
    if alignment["extra_columns"]:
        logger.info("Extra input columns ignored: %s", len(alignment["extra_columns"]))
    return output
