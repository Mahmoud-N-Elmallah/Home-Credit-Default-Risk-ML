from copy import deepcopy

import joblib
import numpy as np
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from src.common.artifacts import model_artifact_path
from src.model_training.evaluation import calculate_metric, save_evaluation_report, save_feature_importance
from src.model_training.models import fit_model, get_imbalance_sampler
from src.model_training.preprocessing import TrainingPreprocessor


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
    save_feature_importance(model, estimator_config["name"], X_processed.columns.to_list(), models_dir, config)
    return model, preprocessor, accelerator


def run_single_search(X, y, ids, config, models_dir, seed, est_config):
    t_config = config["training"]
    name = est_config["name"]
    search_space = est_config.get("search_space", {})
    subsample_rate = t_config["optuna_subsample_rate"]

    print(f"Search phase: {name}, {t_config['optuna_n_trials']} trials, {subsample_rate * 100:.1f}% data.")
    if subsample_rate >= 1.0:
        X_search, y_search, ids_search = X, y, ids
    else:
        X_search, _, y_search, _, ids_search, _ = train_test_split(
            X,
            y,
            ids,
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
    threshold_info = save_evaluation_report(
        y_search,
        search_preds,
        f"{name}_search",
        models_dir,
        config,
        "search_subsample_cv",
        ids=ids_search,
    )
    return best_config, threshold_info
