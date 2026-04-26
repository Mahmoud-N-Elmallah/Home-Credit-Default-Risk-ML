from pathlib import Path
import logging

import polars as pl
import yaml

from src.data_processing.constants import SCHEMA_OVERRIDES


logger = logging.getLogger(__name__)

POLARS_DTYPES = {
    "Int8": pl.Int8,
    "Int16": pl.Int16,
    "Int32": pl.Int32,
    "Int64": pl.Int64,
    "UInt8": pl.UInt8,
    "UInt16": pl.UInt16,
    "UInt32": pl.UInt32,
    "UInt64": pl.UInt64,
    "Float32": pl.Float32,
    "Float64": pl.Float64,
    "String": pl.String,
    "Boolean": pl.Boolean,
}


def _polars_dtype(dtype_name: str):
    try:
        return POLARS_DTYPES[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Polars dtype in config: {dtype_name}") from exc


def _csv_options(config):
    csv_config = config["data"]["csv"]
    schema_overrides = {
        col: _polars_dtype(dtype_name)
        for col, dtype_name in SCHEMA_OVERRIDES.items()
    }
    return {
        "infer_schema_length": csv_config["infer_schema_length"],
        "schema_overrides": schema_overrides,
        "null_values": csv_config["null_values"],
        "ignore_errors": csv_config["ignore_errors"],
    }


def load_data(raw_paths, config):
    """Step 0 - Load & organize source files."""
    logger.info("Loading data...")
    csv_options = _csv_options(config)

    train_base = pl.read_csv(raw_paths["application_train"], **csv_options)
    test_base = pl.read_csv(raw_paths["application_test"], **csv_options)

    bureau = pl.scan_csv(raw_paths["bureau"], **csv_options)
    bureau_balance = pl.scan_csv(raw_paths["bureau_balance"], **csv_options)
    prev_app = pl.scan_csv(raw_paths["previous_application"], **csv_options)
    pos_cash = pl.scan_csv(raw_paths["pos_cash_balance"], **csv_options)
    installments = pl.scan_csv(raw_paths["installments_payments"], **csv_options)
    cc_balance = pl.scan_csv(raw_paths["credit_card_balance"], **csv_options)

    return train_base, test_base, bureau, bureau_balance, prev_app, pos_cash, installments, cc_balance


def write_feature_manifest(train: pl.DataFrame, test: pl.DataFrame, config, cleanup_info):
    final_paths = config["data"]["final"]
    manifest_path = Path(final_paths["feature_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    target_col = config["training"]["target_col"]
    id_col = config["training"]["id_col"]
    feature_cols = [col for col in train.columns if col not in [target_col, id_col]]
    manifest = {
        "enabled_feature_sets": config["pipeline"]["feature_engineering"]["enabled_sets"],
        "source_tables_used": sorted(config["data"]["raw"].keys()),
        "final_train_shape": list(train.shape),
        "final_test_shape": list(test.shape),
        "final_feature_count": len(feature_cols),
        "train_columns": train.columns,
        "test_columns": test.columns,
        "dropped_columns": cleanup_info,
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)
    logger.info("Feature manifest saved to %s", manifest_path)


def latest_submission_path(config):
    artifact_paths = config["training"]["artifact_paths"]
    models_dir = Path(artifact_paths["models_dir"])
    latest_path = Path(artifact_paths.get("latest_experiment", "latest_experiment.txt"))
    if not latest_path.is_absolute():
        latest_path = models_dir / latest_path
    if not latest_path.exists():
        return None

    experiment_dir = Path(latest_path.read_text(encoding="utf-8").strip())
    submission_path = Path(artifact_paths["submission"])
    return submission_path if submission_path.is_absolute() else experiment_dir / submission_path
