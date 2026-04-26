from pathlib import Path
import logging

import joblib
import optuna
import pandas as pd

from src.common.artifacts import model_artifact_path
from src.common.logging import configure_logging
from src.common.schema import clean_column_names
from src.model_training.artifacts import (
    build_run_metadata,
    create_experiment_dir,
    save_config_snapshot,
    validate_reusable_artifacts,
    write_latest_experiment_pointer,
    write_run_metadata,
)
from src.model_training.config import (
    get_estimator_config_by_name,
    get_primary_estimator_config,
    primary_model_name,
)
from src.model_training.evaluation import save_evaluation_report
from src.model_training.models import ACCELERATOR_CACHE
from src.model_training.preprocessing import TrainingPreprocessor
from src.model_training.search import cross_validated_single_predictions, fit_final_single_model, run_single_search


logger = logging.getLogger(__name__)


def configure_optuna_logging(config):
    level_name = str(config["training"]["verbosity"].get("optuna", "INFO")).upper()
    levels = {
        "DEBUG": optuna.logging.DEBUG,
        "INFO": optuna.logging.INFO,
        "WARNING": optuna.logging.WARNING,
        "ERROR": optuna.logging.ERROR,
        "CRITICAL": optuna.logging.CRITICAL,
    }
    optuna.logging.set_verbosity(levels.get(level_name, optuna.logging.INFO))


def run_training(config):
    logger.info("Initializing training pipeline...")
    t_config = config["training"]
    seed = config["globals"]["random_state"]
    models_root = Path(t_config["artifact_paths"]["models_dir"])
    models_root.mkdir(parents=True, exist_ok=True)
    models_dir, experiment_id, timestamp = create_experiment_dir(models_root, config)
    training_log = configure_logging(models_dir / "logs", "training.log")
    save_config_snapshot(models_dir, config)
    logger.info("Experiment artifacts: %s", models_dir)
    logger.info("Training log: %s", training_log)
    configure_optuna_logging(config)

    train_path = Path(config["data"]["final"]["train"])
    test_path = Path(config["data"]["final"]["test"])
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}. Run run.step=process first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}. Run run.step=process first.")

    df = pd.read_csv(train_path)
    train_ids = df[t_config["id_col"]]
    y = df[t_config["target_col"]]
    X = clean_column_names(df.drop(columns=[t_config["target_col"], t_config["id_col"]]))
    metadata = build_run_metadata(config, X, y, train_path, experiment_id, timestamp)
    validate_reusable_artifacts(models_dir, config, metadata)

    run_single_phases(X, y, train_ids, config, models_dir, seed, metadata)

    metadata["selected_accelerators"].update(
        {f"{name}:{mode}": acc for (name, mode), acc in ACCELERATOR_CACHE.items()}
    )
    write_run_metadata(models_dir, config, metadata)
    write_latest_experiment_pointer(models_root, config, models_dir)


def run_single_phases(X, y, ids, config, models_dir, seed, metadata):
    t_config = config["training"]
    phases = t_config["phases"]
    est_config = get_primary_estimator_config(config)
    name = est_config["name"]
    best_config = est_config

    if phases["search"]:
        best_config, search_threshold = run_single_search(X, y, ids, config, models_dir, seed, est_config)
        metadata["chosen_threshold"] = search_threshold
    if phases["validate"] and t_config["run_full_oof_validation"]:
        report_name = f"{name}_optuna"
        logger.info("Scoring best model with full-data OOF validation...")
        oof_preds = cross_validated_single_predictions(
            X,
            y,
            best_config,
            config,
            desc=f"{report_name} OOF",
            is_trial=False,
            feature_selection_enabled=True,
        )
        metadata["chosen_threshold"] = save_evaluation_report(
            y,
            oof_preds,
            report_name,
            models_dir,
            config,
            "out_of_fold",
            ids=ids,
        )
    elif phases["validate"]:
        logger.info("Skipping full-data OOF validation by run profile.")

    if phases["final_fit"]:
        model, preprocessor, accelerator = fit_final_single_model(X, y, best_config, config, models_dir, name)
        metadata["selected_accelerators"][name] = accelerator
        predict_test_and_submit(model, config, models_dir, preprocessor=preprocessor)


def predict_test_and_submit(model_obj, config, models_dir, preprocessor=None):
    logger.info("Generating predictions...")
    test_path = Path(config["data"]["final"]["test"])
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}. Run run.step=process first.")

    df_test = pd.read_csv(test_path)
    id_col = config["training"]["id_col"]
    target_col = config["training"]["target_col"]
    if target_col in df_test.columns:
        raise ValueError(f"{target_col} found in test data.")

    ids = df_test[id_col]
    X_test = clean_column_names(df_test.drop(columns=[id_col]))

    if preprocessor is None:
        preprocessor_path = model_artifact_path(models_dir, config, "preprocessor")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}.")
        preprocessor = joblib.load(preprocessor_path)

    X_test = preprocessor.transform(X_test)

    preds = model_obj.predict_proba(X_test)[:, 1]

    submission_path = model_artifact_path(models_dir, config, "submission")
    submission = pd.DataFrame({id_col: ids, target_col: preds})
    if len(submission) != len(df_test):
        raise ValueError("Submission row count does not match test row count.")
    submission.to_csv(submission_path, index=False)
    logger.info("Submission saved to %s", submission_path)
