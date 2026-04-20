from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import yaml
from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier, early_stopping
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier


SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}

SAMPLERS = {
    "smote": SMOTE,
    "borderline_smote": BorderlineSMOTE,
    "adasyn": ADASYN,
    "random_undersample": RandomUnderSampler,
}

MODELS = {
    "catboost": CatBoostClassifier,
    "lightgbm": LGBMClassifier,
    "xgboost": XGBClassifier,
}

ACCELERATOR_CACHE = {}


def resolve_training_config(config):
    resolved = deepcopy(config)
    training_config = resolved["training"]
    profile_name = training_config["run_mode"]
    profiles = training_config.get("run_profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown training.run_mode: {profile_name}")

    profile = profiles[profile_name]
    for key in ["cv_splits", "optuna_n_trials", "optuna_subsample_rate"]:
        training_config[key] = profile[key]

    feature_selection = training_config["preprocessing"]["feature_selection"]
    feature_selection["enabled_during_search"] = profile["feature_selection_enabled_during_search"]
    training_config["run_full_oof_validation"] = profile["run_full_oof_validation"]
    return resolved


def stable_yaml_hash(data):
    dumped = yaml.safe_dump(data, sort_keys=True)
    return sha256(dumped.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_path(models_dir, config):
    metadata_name = config["training"]["artifact_reuse"]["metadata"]
    path = Path(metadata_name)
    return path if path.is_absolute() else models_dir / path


def build_run_metadata(config, X, y, train_path):
    phases = config["training"]["phases"]
    metric_scopes = []
    if phases["search"]:
        metric_scopes.append("search_subsample_cv")
    if phases["validate"] and config["training"]["run_full_oof_validation"]:
        metric_scopes.append("out_of_fold")
    if phases["final_fit"]:
        metric_scopes.append("final_train_fit")

    return {
        "config_hash": stable_yaml_hash(config),
        "data_hashes": {str(train_path): file_hash(train_path)},
        "run_mode": config["training"]["run_mode"],
        "phases": phases,
        "cv_splits": config["training"]["cv_splits"],
        "optuna_n_trials": config["training"]["optuna_n_trials"],
        "optuna_subsample_rate": config["training"]["optuna_subsample_rate"],
        "row_count": int(len(X)),
        "feature_count": int(X.shape[1]),
        "positive_count": int(y.sum()),
        "metric_scopes": metric_scopes,
        "selected_accelerators": {},
    }


def write_run_metadata(models_dir, config, metadata):
    path = metadata_path(models_dir, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)


def validate_reusable_artifacts(models_dir, config, metadata):
    if config["training"]["artifact_reuse"]["allow_stale_artifacts"]:
        return
    path = metadata_path(models_dir, config)
    if not path.exists():
        return
    existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if existing.get("config_hash") != metadata["config_hash"]:
        raise ValueError("Existing training artifacts use different config. Set allow_stale_artifacts=true to reuse.")
    if existing.get("data_hashes") != metadata["data_hashes"]:
        raise ValueError("Existing training artifacts use different data. Set allow_stale_artifacts=true to reuse.")


def clean_column_names(df):
    cleaned = [re.sub(r"[^\w]", "_", col) for col in df.columns]
    duplicates = sorted({col for col in cleaned if cleaned.count(col) > 1})
    if duplicates:
        raise ValueError(f"Column-name cleanup produced duplicates: {duplicates[:10]}")
    df = df.copy()
    df.columns = cleaned
    return df


def get_scaler(scaler_name):
    if scaler_name in [None, "none"]:
        return None
    try:
        return SCALERS[scaler_name]()
    except KeyError as exc:
        raise ValueError(f"Unknown scaler in config: {scaler_name}") from exc


def parse_selector_threshold(value):
    if isinstance(value, str) and value.lower() == "-inf":
        return -np.inf
    return value


def get_acceleration_config(config):
    return config["training"].get("acceleration", {})


def get_accelerator_order(config):
    acceleration_config = get_acceleration_config(config)
    preferred = acceleration_config.get("preferred", "cpu")
    fallback = acceleration_config.get("fallback", "cpu")
    retry = acceleration_config.get("retry_on_failure", True)
    if retry and fallback != preferred:
        return [preferred, fallback]
    return [preferred]


def get_accelerator_params(model_name, config, accelerator):
    acceleration_config = get_acceleration_config(config)
    model_config = acceleration_config.get("models", {}).get(model_name, {})
    return model_config.get(f"{accelerator}_params", {}).copy()


def accelerator_failure_is_retryable(error, config):
    message = str(error).lower()
    keywords = get_acceleration_config(config).get("retry_failure_keywords", [])
    return any(keyword.lower() in message for keyword in keywords)


def get_imbalance_config(config):
    return config["training"]["preprocessing"]["imbalance"]


def get_imbalance_sampler(config):
    imbalance_config = get_imbalance_config(config)
    strategy = imbalance_config["strategy"]
    if strategy in [None, "none", "class_weight"]:
        return None
    try:
        sampler_cls = SAMPLERS[strategy]
    except KeyError as exc:
        raise ValueError(f"Unknown imbalance strategy in config: {strategy}") from exc
    params = imbalance_config.get("sampler_params", {}).copy()
    params.setdefault("random_state", config["globals"]["random_state"])
    return sampler_cls(**params)


def split_speed_params(params, model_name=None):
    model_params = params.copy() if params else {}
    fit_options = {
        "eval_fraction": model_params.pop("eval_fraction", None),
        "early_stopping_rounds": model_params.get("early_stopping_rounds"),
    }
    if model_name != "xgboost":
        model_params.pop("early_stopping_rounds", None)
    return model_params, fit_options


def merge_model_params(name, params, config, is_trial=True, accelerator=None):
    base_params, _ = split_speed_params(params, name)
    model_params = base_params.copy()
    if accelerator is not None:
        model_params.update(get_accelerator_params(name, config, accelerator))
    model_params.setdefault("random_state", config["globals"]["random_state"])

    imbalance_config = get_imbalance_config(config)
    if imbalance_config["strategy"] == "class_weight":
        model_params.update(imbalance_config.get("class_weight_params", {}).get(name, {}))

    if name == "catboost":
        verbosity = config["training"]["verbosity"]
        verbosity_key = "catboost_trial" if is_trial else "catboost_final"
        model_params.setdefault("verbose", verbosity.get(verbosity_key, 0))

    return model_params


def fit_kwargs_for_model(name, fit_options, X, y, config):
    eval_fraction = fit_options.get("eval_fraction")
    early_stopping_rounds = fit_options.get("early_stopping_rounds")
    if not eval_fraction or not early_stopping_rounds or len(X) < 100:
        return X, y, {}

    X_fit, X_eval, y_fit, y_eval = train_test_split(
        X,
        y,
        test_size=eval_fraction,
        stratify=y,
        random_state=config["globals"]["random_state"],
    )
    if name == "catboost":
        return X_fit, y_fit, {"eval_set": (X_eval, y_eval), "early_stopping_rounds": early_stopping_rounds}
    if name == "lightgbm":
        return X_fit, y_fit, {
            "eval_set": [(X_eval, y_eval)],
            "callbacks": [early_stopping(early_stopping_rounds, verbose=False)],
        }
    if name == "xgboost":
        return X_fit, y_fit, {"eval_set": [(X_eval, y_eval)], "verbose": False}
    return X, y, {}


def capability_sample(X, y, config):
    if len(X) <= 512:
        return X, y
    sample_size = min(len(X), 512)
    _, X_sample, _, y_sample = train_test_split(
        X,
        y,
        test_size=sample_size,
        stratify=y,
        random_state=config["globals"]["random_state"],
    )
    return X_sample, y_sample


def resolve_model_accelerator(name, params, config, X, y, is_trial):
    cache_key = (name, config["training"]["run_mode"])
    if cache_key in ACCELERATOR_CACHE:
        return ACCELERATOR_CACHE[cache_key]

    X_sample, y_sample = capability_sample(X, y, config)
    params_no_speed, _ = split_speed_params(params, name)
    test_params = params_no_speed.copy()
    if name == "catboost":
        test_params["iterations"] = min(int(test_params.get("iterations", 10)), 2)
    elif name in ["lightgbm", "xgboost"]:
        test_params["n_estimators"] = min(int(test_params.get("n_estimators", 10)), 2)

    accelerators = get_accelerator_order(config)
    last_error = None
    for accelerator in accelerators:
        model_params = merge_model_params(name, test_params, config, is_trial=is_trial, accelerator=accelerator)
        try:
            MODELS[name](**model_params).fit(X_sample, y_sample)
            ACCELERATOR_CACHE[cache_key] = accelerator
            return accelerator
        except Exception as error:
            last_error = error
            if accelerator != accelerators[-1] and accelerator_failure_is_retryable(error, config):
                continue
            raise
    raise last_error


def fit_model(name, params, config, X, y, is_trial):
    accelerator = resolve_model_accelerator(name, params, config, X, y, is_trial)
    model_params = merge_model_params(name, params, config, is_trial=is_trial, accelerator=accelerator)
    _, fit_options = split_speed_params(params, name)
    X_fit, y_fit, fit_kwargs = fit_kwargs_for_model(name, fit_options, X, y, config)
    model = MODELS[name](**model_params)
    try:
        model.fit(X_fit, y_fit, **fit_kwargs)
    except Exception as error:
        if accelerator_failure_is_retryable(error, config) and accelerator != get_accelerator_order(config)[-1]:
            raise RuntimeError(f"{name} failed after cached accelerator selection. Error was: {error}") from error
        raise
    return model, accelerator


class TrainingPreprocessor:
    def __init__(self, config, feature_selection_enabled=True):
        self.config = config
        self.feature_selection_enabled = feature_selection_enabled
        self.scaler = None
        self.selector = None
        self.selected_columns = None

    def fit(self, X, y):
        t_config = self.config["training"]
        seed = self.config["globals"]["random_state"]
        X_work = X.copy()

        scaler_name = t_config["preprocessing"]["scaler"]
        self.scaler = get_scaler(scaler_name)
        if self.scaler is not None:
            X_work = pd.DataFrame(self.scaler.fit_transform(X_work), columns=X_work.columns, index=X_work.index)

        fs_config = t_config["preprocessing"]["feature_selection"]
        if self.feature_selection_enabled and fs_config["method"] == "lgbm":
            selector_params = fs_config.get("selector_params", {}).copy()
            self.selector = fit_lgbm_selector(X_work, y, selector_params, fs_config, self.config, seed)
            self.selected_columns = X_work.columns[self.selector.get_support()].to_list()
        elif fs_config["method"] not in [None, "none", "lgbm"]:
            raise ValueError(f"Unknown feature_selection.method: {fs_config['method']}")

        return self

    def transform(self, X):
        X_work = X.copy()
        if self.scaler is not None:
            X_work = pd.DataFrame(self.scaler.transform(X_work), columns=X_work.columns, index=X_work.index)
        if self.selected_columns is not None:
            missing_cols = [col for col in self.selected_columns if col not in X_work.columns]
            if missing_cols:
                raise ValueError(f"Missing selected columns during transform: {missing_cols[:10]}")
            X_work = X_work[self.selected_columns]
        return X_work

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def fit_lgbm_selector(X, y, selector_params, fs_config, config, seed):
    accelerator = resolve_model_accelerator("lightgbm", selector_params, config, X, y, is_trial=True)
    model_params = selector_params.copy()
    model_params.update(get_accelerator_params("lightgbm", config, accelerator))
    model_params.setdefault("random_state", seed)
    selector = SelectFromModel(
        LGBMClassifier(**model_params),
        max_features=fs_config["max_features"],
        threshold=parse_selector_threshold(fs_config["threshold"]),
    )
    selector.fit(X, y)
    return selector


def calculate_metric(y_true, y_pred_prob, metric_name, threshold):
    if metric_name == "roc_auc":
        return roc_auc_score(y_true, y_pred_prob)
    if metric_name == "average_precision":
        return average_precision_score(y_true, y_pred_prob)
    if metric_name == "f1":
        return f1_score(y_true, (y_pred_prob > threshold).astype(int))
    raise ValueError(f"Unknown metric: {metric_name}")


def model_artifact_path(models_dir, config, key, **kwargs):
    pattern = config["training"]["artifact_paths"][key]
    path = Path(pattern.format(**kwargs))
    return path if path.is_absolute() else models_dir / path


def save_evaluation_report(y_true, y_pred_prob, model_name, models_dir, config, evaluation_scope):
    allowed_scopes = {"out_of_fold", "search_subsample_cv", "final_train_fit"}
    if evaluation_scope not in allowed_scopes:
        raise ValueError(f"Invalid evaluation scope: {evaluation_scope}")

    t_config = config["training"]
    eval_config = t_config["evaluation"]
    threshold = t_config["classification_threshold"]
    y_pred_bin = (y_pred_prob > threshold).astype(int)

    report_str = f"Model: {model_name}\n"
    report_str += "=" * 40 + "\n"
    report_str += f"Evaluation Scope: {evaluation_scope}\n"
    report_str += f"Run Mode: {t_config['run_mode']}\n"
    report_str += f"Classification Threshold: {threshold}\n"
    report_str += f"ROC AUC Score: {roc_auc_score(y_true, y_pred_prob):.4f}\n"
    report_str += f"Average Precision (PR AUC): {average_precision_score(y_true, y_pred_prob):.4f}\n"
    report_str += f"F1 Score: {f1_score(y_true, y_pred_bin):.4f}\n\n"
    report_str += "Classification Report:\n"
    report_str += classification_report(y_true, y_pred_bin, digits=eval_config["report_digits"], zero_division=0)

    models_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_path(models_dir, config, "evaluation_report", model_name=model_name).write_text(
        report_str,
        encoding="utf-8",
    )

    cm = confusion_matrix(y_true, y_pred_bin)
    plt.figure(figsize=tuple(eval_config["confusion_matrix_figsize"]))
    sns.heatmap(cm, annot=True, fmt="d", cmap=eval_config["confusion_matrix_cmap"])
    plt.title(f"Confusion Matrix: {model_name} ({evaluation_scope})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(model_artifact_path(models_dir, config, "confusion_matrix", model_name=model_name))
    plt.close()


def suggest_params(trial, search_space):
    params = {}
    for param_name, space in search_space.items():
        if space["type"] == "int":
            params[param_name] = trial.suggest_int(param_name, space["low"], space["high"], log=space.get("log", False))
        elif space["type"] == "float":
            params[param_name] = trial.suggest_float(param_name, space["low"], space["high"], log=space.get("log", False))
        elif space["type"] == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, space["choices"])
        else:
            raise ValueError(f"Unknown search space type for {param_name}: {space['type']}")
    return params


def merged_estimator_config(est_config, param_overrides=None):
    candidate = deepcopy(est_config)
    params = est_config.get("params", {}).copy()
    if param_overrides:
        params.update(param_overrides)
    candidate["params"] = params
    return candidate


def get_cv(config):
    t_config = config["training"]
    return StratifiedKFold(
        n_splits=t_config["cv_splits"],
        shuffle=t_config["cv_shuffle"],
        random_state=config["globals"]["random_state"],
    )


def fit_predict_fold(X_train, y_train, X_valid, estimator_config, config, is_trial, feature_selection_enabled):
    preprocessor = TrainingPreprocessor(config, feature_selection_enabled=feature_selection_enabled)
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_valid_processed = preprocessor.transform(X_valid)

    sampler = get_imbalance_sampler(config)
    if sampler is not None:
        X_train_processed, y_train = sampler.fit_resample(X_train_processed, y_train)

    model, _ = fit_model(
        estimator_config["name"],
        estimator_config.get("params", {}),
        config,
        X_train_processed,
        y_train,
        is_trial,
    )
    return model.predict_proba(X_valid_processed)[:, 1]


def cross_validated_single_predictions(X, y, estimator_config, config, desc, is_trial, feature_selection_enabled):
    cv = get_cv(config)
    oof_preds = np.full(len(X), np.nan)

    for train_idx, valid_idx in tqdm(cv.split(X, y), total=config["training"]["cv_splits"], desc=desc):
        oof_preds[valid_idx] = fit_predict_fold(
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[valid_idx],
            estimator_config,
            config,
            is_trial=is_trial,
            feature_selection_enabled=feature_selection_enabled,
        )

    if np.isnan(oof_preds).any():
        raise ValueError("OOF predictions incomplete; at least one row was not predicted.")
    return oof_preds


def cross_validated_ensemble_predictions(X, y, estimators, config):
    cv = get_cv(config)
    oof_preds = np.full(len(X), np.nan)
    total_weight = sum(estimator["weight"] for estimator in estimators)

    for train_idx, valid_idx in tqdm(cv.split(X, y), total=config["training"]["cv_splits"], desc="Ensemble Folds"):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        preprocessor = TrainingPreprocessor(config, feature_selection_enabled=True)
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_valid_processed = preprocessor.transform(X_valid)

        sampler = get_imbalance_sampler(config)
        if sampler is not None:
            X_train_processed, y_train = sampler.fit_resample(X_train_processed, y_train)

        fold_preds = np.zeros(len(X_valid))
        for estimator_config in estimators:
            model, _ = fit_model(
                estimator_config["name"],
                estimator_config.get("params", {}),
                config,
                X_train_processed,
                y_train,
                False,
            )
            fold_preds += model.predict_proba(X_valid_processed)[:, 1] * (estimator_config["weight"] / total_weight)
        oof_preds[valid_idx] = fold_preds

    if np.isnan(oof_preds).any():
        raise ValueError("OOF predictions incomplete; at least one row was not predicted.")
    return oof_preds


def fit_final_single_model(X, y, estimator_config, config, models_dir, model_name):
    preprocessor = TrainingPreprocessor(config, feature_selection_enabled=True)
    X_processed = preprocessor.fit_transform(X, y)

    sampler = get_imbalance_sampler(config)
    y_fit = y
    if sampler is not None:
        X_processed, y_fit = sampler.fit_resample(X_processed, y_fit)

    model, accelerator = fit_model(
        estimator_config["name"],
        estimator_config.get("params", {}),
        config,
        X_processed,
        y_fit,
        False,
    )
    joblib.dump(preprocessor, model_artifact_path(models_dir, config, "preprocessor"))
    joblib.dump(model, model_artifact_path(models_dir, config, "single_model", model_name=model_name))
    return model, preprocessor, accelerator


def fit_final_ensemble(X, y, estimators, config, models_dir):
    preprocessor = TrainingPreprocessor(config, feature_selection_enabled=True)
    X_processed = preprocessor.fit_transform(X, y)

    sampler = get_imbalance_sampler(config)
    y_fit = y
    if sampler is not None:
        X_processed, y_fit = sampler.fit_resample(X_processed, y_fit)

    joblib.dump(preprocessor, model_artifact_path(models_dir, config, "preprocessor"))

    final_models = []
    accelerators = {}
    for estimator_config in estimators:
        model, accelerator = fit_model(
            estimator_config["name"],
            estimator_config.get("params", {}),
            config,
            X_processed,
            y_fit,
            False,
        )
        joblib.dump(
            model,
            model_artifact_path(models_dir, config, "ensemble_model", model_name=estimator_config["name"]),
        )
        final_models.append((model, estimator_config["weight"]))
        accelerators[estimator_config["name"]] = accelerator

    return final_models, preprocessor, accelerators


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
    config = resolve_training_config(config)
    print(f"Initializing training pipeline ({config['training']['run_mode']})...")
    t_config = config["training"]
    seed = config["globals"]["random_state"]
    models_dir = Path(t_config["artifact_paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    configure_optuna_logging(config)

    train_path = Path(config["data"]["final"]["train"])
    test_path = Path(config["data"]["final"]["test"])
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}. Run --process first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}. Run --process first.")

    df = pd.read_csv(train_path)
    y = df[t_config["target_col"]]
    X = clean_column_names(df.drop(columns=[t_config["target_col"], t_config["id_col"]]))
    metadata = build_run_metadata(config, X, y, train_path)
    validate_reusable_artifacts(models_dir, config, metadata)

    model_type = t_config["models"]["type"]
    if model_type == "single":
        run_single_phases(X, y, config, models_dir, seed, metadata)
    elif model_type == "ensemble":
        run_ensemble_phases(X, y, config, models_dir, metadata)
    else:
        raise ValueError(f"Unknown training.models.type: {model_type}")

    metadata["selected_accelerators"] = {f"{name}:{mode}": acc for (name, mode), acc in ACCELERATOR_CACHE.items()}
    write_run_metadata(models_dir, config, metadata)


def run_single_phases(X, y, config, models_dir, seed, metadata):
    t_config = config["training"]
    phases = t_config["phases"]
    est_config = t_config["models"]["estimators"][0]
    name = est_config["name"]
    best_config = est_config

    if phases["search"]:
        best_config = run_single_search(X, y, config, models_dir, seed, est_config)
    if phases["validate"] and t_config["run_full_oof_validation"]:
        report_name = f"{name}_optuna"
        print("Scoring best model with full-data OOF validation...")
        oof_preds = cross_validated_single_predictions(
            X,
            y,
            best_config,
            config,
            desc=f"{report_name} OOF",
            is_trial=False,
            feature_selection_enabled=True,
        )
        save_evaluation_report(y, oof_preds, report_name, models_dir, config, "out_of_fold")
    elif phases["validate"]:
        print("Skipping full-data OOF validation by run profile.")

    if phases["final_fit"]:
        model, preprocessor, accelerator = fit_final_single_model(X, y, best_config, config, models_dir, name)
        metadata["selected_accelerators"][name] = accelerator
        predict_test_and_submit(model, config, preprocessor=preprocessor, is_ensemble=False)


def run_single_search(X, y, config, models_dir, seed, est_config):
    t_config = config["training"]
    name = est_config["name"]
    search_space = est_config.get("search_space", {})
    subsample_rate = t_config["optuna_subsample_rate"]

    print(f"Search phase: {name}, {t_config['optuna_n_trials']} trials, {subsample_rate * 100:.1f}% data.")
    X_search, _, y_search, _ = train_test_split(
        X,
        y,
        train_size=subsample_rate,
        stratify=y,
        random_state=seed,
    )
    feature_selection_enabled = t_config["preprocessing"]["feature_selection"]["enabled_during_search"]

    def objective(trial):
        trial_params = suggest_params(trial, search_space)
        model_params = {key.replace("model__", ""): value for key, value in trial_params.items()}
        candidate_config = merged_estimator_config(est_config, model_params)
        preds = cross_validated_single_predictions(
            X_search,
            y_search,
            candidate_config,
            config,
            desc=f"{name} Search CV",
            is_trial=True,
            feature_selection_enabled=feature_selection_enabled,
        )
        return calculate_metric(y_search, preds, t_config["optimization_metric"], t_config["classification_threshold"])

    study = optuna.create_study(
        direction=t_config["optuna_direction"],
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=t_config["optuna_n_trials"], show_progress_bar=True)
    best_params = {key.replace("model__", ""): value for key, value in study.best_params.items()}
    with open(model_artifact_path(models_dir, config, "best_params", model_name=name), "w", encoding="utf-8") as file:
        yaml.safe_dump(best_params, file)

    best_config = merged_estimator_config(est_config, best_params)
    search_preds = cross_validated_single_predictions(
        X_search,
        y_search,
        best_config,
        config,
        desc=f"{name} Search Best CV",
        is_trial=True,
        feature_selection_enabled=feature_selection_enabled,
    )
    save_evaluation_report(y_search, search_preds, f"{name}_search", models_dir, config, "search_subsample_cv")
    return best_config


def run_ensemble_phases(X, y, config, models_dir, metadata):
    phases = config["training"]["phases"]
    estimators = config["training"]["models"]["estimators"]

    if phases["validate"] and config["training"]["run_full_oof_validation"]:
        print("Scoring ensemble with full-data OOF validation...")
        oof_preds = cross_validated_ensemble_predictions(X, y, estimators, config)
        save_evaluation_report(y, oof_preds, "ensemble_cv_oof", models_dir, config, "out_of_fold")
    elif phases["validate"]:
        print("Skipping full-data OOF validation by run profile.")

    if phases["final_fit"]:
        final_models, preprocessor, accelerators = fit_final_ensemble(X, y, estimators, config, models_dir)
        metadata["selected_accelerators"].update(accelerators)
        predict_test_and_submit(final_models, config, preprocessor=preprocessor, is_ensemble=True)


def predict_test_and_submit(model_obj, config, preprocessor=None, is_ensemble=False):
    print("Generating predictions...")
    test_path = Path(config["data"]["final"]["test"])
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}. Run --process first.")

    df_test = pd.read_csv(test_path)
    id_col = config["training"]["id_col"]
    target_col = config["training"]["target_col"]
    if target_col in df_test.columns:
        raise ValueError(f"{target_col} found in test data.")

    ids = df_test[id_col]
    X_test = clean_column_names(df_test.drop(columns=[id_col]))

    models_dir = Path(config["training"]["artifact_paths"]["models_dir"])
    if preprocessor is None:
        preprocessor_path = model_artifact_path(models_dir, config, "preprocessor")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}.")
        preprocessor = joblib.load(preprocessor_path)

    X_test = preprocessor.transform(X_test)

    if not is_ensemble:
        preds = model_obj.predict_proba(X_test)[:, 1]
    else:
        preds = np.zeros(len(X_test))
        total_weight = sum(weight for _, weight in model_obj)
        for model, weight in model_obj:
            preds += model.predict_proba(X_test)[:, 1] * (weight / total_weight)

    submission_path = Path(config["training"]["artifact_paths"]["submission"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({id_col: ids, target_col: preds})
    if len(submission) != len(df_test):
        raise ValueError("Submission row count does not match test row count.")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
