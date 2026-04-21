from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from pathlib import Path
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
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
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


def slugify(value):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return slug.strip("_") or "experiment"


def primary_model_name(config):
    return config["training"]["models"]["primary"]


def get_primary_estimator_config(config):
    primary = primary_model_name(config)
    return get_estimator_config_by_name(config, primary)


def get_estimator_config_by_name(config, name):
    for candidate in config["training"]["models"]["candidates"]:
        if candidate["name"] == name:
            return candidate
    raise ValueError(f"Model not found in training.models.candidates: {name}")


def create_experiment_dir(models_root, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_config = config["training"]
    exp_config = t_config["experiment"]
    if exp_config.get("name"):
        experiment_id = slugify(exp_config["name"])
    else:
        experiment_id = exp_config["folder_template"].format(
            timestamp=timestamp,
            run_mode=t_config["run_mode"],
            primary_model=primary_model_name(config),
        )
        experiment_id = slugify(experiment_id)

    experiment_dir = models_root / experiment_id
    suffix = 1
    while experiment_dir.exists():
        suffix += 1
        experiment_dir = models_root / f"{experiment_id}_{suffix}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir, experiment_dir.name, timestamp


def write_latest_experiment_pointer(models_root, config, experiment_dir):
    latest_path = Path(config["training"]["artifact_paths"]["latest_experiment"])
    if not latest_path.is_absolute():
        latest_path = models_root / latest_path
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(experiment_dir.resolve()), encoding="utf-8")


def save_config_snapshot(experiment_dir, config):
    path = model_artifact_path(experiment_dir, config, "config_snapshot")
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def metadata_path(models_dir, config):
    metadata_name = config["training"]["artifact_reuse"]["metadata"]
    path = Path(metadata_name)
    return path if path.is_absolute() else models_dir / path


def build_run_metadata(config, X, y, train_path, experiment_id, timestamp):
    phases = config["training"]["phases"]
    metric_scopes = []
    if phases["search"]:
        metric_scopes.append("search_subsample_cv")
    if phases["validate"] and config["training"]["run_full_oof_validation"]:
        metric_scopes.append("out_of_fold")
    if phases["final_fit"]:
        metric_scopes.append("final_train_fit")

    return {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "config_hash": stable_yaml_hash(config),
        "data_hashes": {str(train_path): file_hash(train_path)},
        "run_mode": config["training"]["run_mode"],
        "primary_model": primary_model_name(config),
        "phases": phases,
        "cv_splits": config["training"]["cv_splits"],
        "optuna_n_trials": config["training"]["optuna_n_trials"],
        "optuna_subsample_rate": config["training"]["optuna_subsample_rate"],
        "row_count": int(len(X)),
        "feature_count": int(X.shape[1]),
        "positive_count": int(y.sum()),
        "metric_scopes": metric_scopes,
        "selected_accelerators": {},
        "chosen_threshold": None,
        "artifact_list": [],
    }


def write_run_metadata(models_dir, config, metadata):
    metadata_file_name = config["training"]["artifact_reuse"]["metadata"]
    artifact_list = [
        str(path.relative_to(models_dir)).replace("\\", "/")
        for path in models_dir.rglob("*")
        if path.is_file() and path.name != metadata_file_name
    ]
    path = metadata_path(models_dir, config)
    artifact_list.append(str(path.relative_to(models_dir)).replace("\\", "/"))
    metadata["artifact_list"] = sorted(artifact_list)
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
    elif name == "lightgbm":
        model_params.setdefault("verbosity", config["training"]["verbosity"].get("lgbm", -1))

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
    test_params.pop("early_stopping_rounds", None)
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
        self.pruned_columns = None

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

        pruning_config = t_config["preprocessing"].get("feature_pruning", {})
        if pruning_config.get("enabled", False):
            self.pruned_columns = fit_feature_pruning_columns(
                X_work,
                y,
                self.selected_columns,
                self.selector,
                pruning_config,
            )

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
        if self.pruned_columns is not None:
            missing_cols = [col for col in self.pruned_columns if col not in X_work.columns]
            if missing_cols:
                raise ValueError(f"Missing pruned columns during transform: {missing_cols[:10]}")
            X_work = X_work[self.pruned_columns]
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


def fit_feature_pruning_columns(X, y, selected_columns, selector, pruning_config):
    if pruning_config["source"] != "feature_importance":
        raise ValueError(f"Unknown feature_pruning.source: {pruning_config['source']}")

    candidate_columns = selected_columns or X.columns.to_list()
    if selector is not None and hasattr(selector, "estimator_") and hasattr(selector.estimator_, "feature_importances_"):
        importances = pd.Series(selector.estimator_.feature_importances_, index=X.columns)
    else:
        importances = pd.Series(np.ones(len(X.columns)), index=X.columns)

    scores = importances.reindex(candidate_columns).fillna(0.0)
    min_importance = float(pruning_config["min_importance"])
    keep_columns = scores[scores >= min_importance].sort_values(ascending=False).index.to_list()

    keep_top_n = pruning_config.get("keep_top_n")
    if keep_top_n is not None:
        keep_columns = keep_columns[: int(keep_top_n)]

    always_keep = [col for col in pruning_config.get("always_keep", []) if col in candidate_columns]
    keep_columns = list(dict.fromkeys(always_keep + keep_columns))
    if not keep_columns:
        raise ValueError("Feature pruning removed all columns. Relax feature_pruning thresholds.")
    return keep_columns


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
    resolved = path if path.is_absolute() else models_dir / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def model_feature_importances(model, model_name, feature_names):
    if model_name == "catboost":
        values = model.get_feature_importance()
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        raise ValueError(f"Feature importance is not available for model: {model_name}")

    if len(values) != len(feature_names):
        raise ValueError("Feature importance length does not match processed feature count.")
    frame = pd.DataFrame({"feature": feature_names, "importance": values})
    frame["importance"] = frame["importance"].astype(float)
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def save_feature_importance(model, model_name, feature_names, models_dir, config):
    reports_config = config["training"]["reports"]
    if not reports_config.get("save_feature_importance", False):
        return

    frame = model_feature_importances(model, model_name, feature_names)
    frame.to_csv(model_artifact_path(models_dir, config, "feature_importance"), index=False)

    top_n = int(reports_config["feature_importance_top_n"])
    top = frame.head(top_n).sort_values("importance", ascending=True)
    if top.empty:
        return
    plt.figure(figsize=(8, max(4, min(12, 0.22 * len(top)))))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top {min(top_n, len(frame))} Feature Importances: {model_name}")
    plt.tight_layout()
    plt.savefig(model_artifact_path(models_dir, config, "feature_importance_plot"))
    plt.close()


def threshold_grid(config):
    grid = config["training"]["threshold_tuning"]["grid"]
    min_value = float(grid["min"])
    max_value = float(grid["max"])
    step = float(grid["step"])
    count = int(round((max_value - min_value) / step)) + 1
    return np.round(np.linspace(min_value, max_value, count), 10)


def build_threshold_table(y_true, y_pred_prob, config):
    rows = []
    for threshold in threshold_grid(config):
        y_pred_bin = (y_pred_prob > threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, y_pred_bin, zero_division=0),
                "recall": recall_score(y_true, y_pred_bin, zero_division=0),
                "f1": f1_score(y_true, y_pred_bin, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred_bin),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(y_true, y_pred_prob, config, evaluation_scope):
    tuning_config = config["training"]["threshold_tuning"]
    if not tuning_config["enabled"]:
        threshold = float(config["training"]["classification_threshold"])
        return {
            "threshold": threshold,
            "objective": "configured",
            "source": "config_classification_threshold",
            "score": None,
        }, build_threshold_table(y_true, y_pred_prob, config)

    if tuning_config["objective"] != "f1":
        raise ValueError(f"Unknown threshold_tuning.objective: {tuning_config['objective']}")

    table = build_threshold_table(y_true, y_pred_prob, config)
    best_row = table.sort_values(["f1", "precision", "threshold"], ascending=[False, False, True]).iloc[0]
    source = "oof_max_f1" if evaluation_scope == "out_of_fold" else f"{evaluation_scope}_max_f1"
    return {
        "threshold": float(best_row["threshold"]),
        "objective": "f1",
        "source": source,
        "score": float(best_row["f1"]),
    }, table


def build_lift_table(y_true, y_pred_prob):
    frame = pd.DataFrame({"target": np.asarray(y_true), "prediction": np.asarray(y_pred_prob)})
    frame = frame.sort_values("prediction", ascending=False).reset_index(drop=True)
    frame["decile"] = np.floor(np.arange(len(frame)) * 10 / len(frame)).astype(int) + 1
    total_defaults = frame["target"].sum()
    lift = (
        frame.groupby("decile", as_index=False)
        .agg(
            row_count=("target", "size"),
            default_count=("target", "sum"),
            default_rate=("target", "mean"),
            min_score=("prediction", "min"),
            max_score=("prediction", "max"),
        )
        .sort_values("decile")
    )
    lift = (
        lift.set_index("decile")
        .reindex(range(1, 11))
        .rename_axis("decile")
        .reset_index()
    )
    for col in ["row_count", "default_count"]:
        lift[col] = lift[col].fillna(0).astype(int)
    lift["default_rate"] = lift["default_rate"].fillna(0.0)
    lift["cumulative_default_count"] = lift["default_count"].cumsum()
    if total_defaults > 0:
        lift["cumulative_default_capture"] = lift["cumulative_default_count"] / total_defaults
    else:
        lift["cumulative_default_capture"] = 0.0
    return lift


def save_oof_predictions(y_true, y_pred_prob, ids, models_dir, config):
    if not config["training"]["reports"]["save_oof_predictions"]:
        return
    id_col = config["training"]["id_col"]
    target_col = config["training"]["target_col"]
    output = pd.DataFrame(
        {
            id_col: ids.to_numpy() if ids is not None else np.arange(len(y_true)),
            target_col: np.asarray(y_true),
            "prediction": np.asarray(y_pred_prob),
        }
    )
    output.to_csv(model_artifact_path(models_dir, config, "oof_predictions"), index=False)


def save_diagnostic_plots(y_true, y_pred_prob, y_pred_bin, lift_table, model_name, models_dir, config, evaluation_scope):
    if not config["training"]["reports"]["save_curves"]:
        return

    eval_config = config["training"]["evaluation"]
    cm = confusion_matrix(y_true, y_pred_bin)
    plt.figure(figsize=tuple(eval_config["confusion_matrix_figsize"]))
    sns.heatmap(cm, annot=True, fmt="d", cmap=eval_config["confusion_matrix_cmap"])
    plt.title(f"Confusion Matrix: {model_name} ({evaluation_scope})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(model_artifact_path(models_dir, config, "confusion_matrix"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.tight_layout()
    plt.savefig(model_artifact_path(models_dir, config, "roc_curve"))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.tight_layout()
    plt.savefig(model_artifact_path(models_dir, config, "pr_curve"))
    plt.close()

    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Default Rate")
    plt.title(f"Calibration Curve: {model_name}")
    plt.tight_layout()
    plt.savefig(model_artifact_path(models_dir, config, "calibration_curve"))
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(lift_table["decile"], lift_table["default_rate"])
    plt.xlabel("Risk Decile (1 = highest risk)")
    plt.ylabel("Default Rate")
    plt.title(f"Lift by Decile: {model_name}")
    plt.tight_layout()
    plt.savefig(model_artifact_path(models_dir, config, "lift_chart"))
    plt.close()


def calibrated_oof_predictions(y_true, y_pred_prob, config):
    calibration_config = config["training"]["calibration"]
    method = calibration_config["method"]
    if method != "isotonic":
        raise ValueError(f"Unknown calibration.method: {method}")

    y_array = np.asarray(y_true)
    pred_array = np.asarray(y_pred_prob)
    min_class_count = int(pd.Series(y_array).value_counts().min())
    n_splits = min(int(calibration_config["cv_splits"]), min_class_count)
    if n_splits < 2:
        raise ValueError("Calibration diagnostics require at least two positive and negative samples.")

    calibrated = np.full(len(pred_array), np.nan)
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=config["training"]["cv_shuffle"],
        random_state=config["globals"]["random_state"],
    )
    for train_idx, valid_idx in cv.split(pred_array.reshape(-1, 1), y_array):
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(pred_array[train_idx], y_array[train_idx])
        calibrated[valid_idx] = calibrator.predict(pred_array[valid_idx])

    if np.isnan(calibrated).any():
        raise ValueError("Calibration diagnostics incomplete; at least one row was not calibrated.")
    return np.clip(calibrated, 0.0, 1.0)


def save_calibration_diagnostics(y_true, y_pred_prob, models_dir, config):
    if not config["training"]["calibration"]["enabled"]:
        return None

    calibrated = calibrated_oof_predictions(y_true, y_pred_prob, config)
    diagnostics = {
        "method": config["training"]["calibration"]["method"],
        "cv_splits": config["training"]["calibration"]["cv_splits"],
        "scope": "out_of_fold_probability_calibration",
        "uncalibrated_brier_score": float(brier_score_loss(y_true, y_pred_prob)),
        "calibrated_brier_score": float(brier_score_loss(y_true, calibrated)),
        "apply_to_submission": config["training"]["calibration"]["apply_to_submission"],
    }
    with open(model_artifact_path(models_dir, config, "calibration_diagnostics"), "w", encoding="utf-8") as file:
        yaml.safe_dump(diagnostics, file, sort_keys=False)
    return diagnostics


def save_evaluation_report(y_true, y_pred_prob, model_name, models_dir, config, evaluation_scope, ids=None):
    allowed_scopes = {"out_of_fold", "search_subsample_cv", "final_train_fit"}
    if evaluation_scope not in allowed_scopes:
        raise ValueError(f"Invalid evaluation scope: {evaluation_scope}")

    t_config = config["training"]
    eval_config = t_config["evaluation"]
    threshold_info, threshold_table = choose_threshold(y_true, y_pred_prob, config, evaluation_scope)
    threshold = threshold_info["threshold"]
    y_pred_bin = (y_pred_prob > threshold).astype(int)
    lift_table = build_lift_table(y_true, y_pred_prob)
    metrics = {
        "model": model_name,
        "evaluation_scope": evaluation_scope,
        "run_mode": t_config["run_mode"],
        "threshold": threshold_info,
        "ranking": {
            "roc_auc": float(roc_auc_score(y_true, y_pred_prob)),
            "average_precision": float(average_precision_score(y_true, y_pred_prob)),
            "brier_score": float(brier_score_loss(y_true, y_pred_prob)),
        },
        "threshold_metrics": {
            "precision": float(precision_score(y_true, y_pred_bin, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_bin, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred_bin, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred_bin)),
        },
    }
    calibration_diagnostics = None
    if evaluation_scope == "out_of_fold":
        calibration_diagnostics = save_calibration_diagnostics(y_true, y_pred_prob, models_dir, config)
        if calibration_diagnostics is not None:
            metrics["calibration"] = calibration_diagnostics

    report_str = f"Model: {model_name}\n"
    report_str += "=" * 40 + "\n"
    report_str += f"Evaluation Scope: {evaluation_scope}\n"
    report_str += f"Run Mode: {t_config['run_mode']}\n"
    report_str += f"Classification Threshold: {threshold:.4f}\n"
    report_str += f"Threshold Source: {threshold_info['source']}\n"
    report_str += f"Threshold Objective: {threshold_info['objective']}\n"
    report_str += f"ROC AUC Score: {metrics['ranking']['roc_auc']:.4f}\n"
    report_str += f"Average Precision (PR AUC): {metrics['ranking']['average_precision']:.4f}\n"
    report_str += f"Brier Score: {metrics['ranking']['brier_score']:.4f}\n"
    if calibration_diagnostics is not None:
        report_str += f"Calibrated Brier Score: {calibration_diagnostics['calibrated_brier_score']:.4f}\n"
    report_str += f"Precision: {metrics['threshold_metrics']['precision']:.4f}\n"
    report_str += f"Recall: {metrics['threshold_metrics']['recall']:.4f}\n"
    report_str += f"F1 Score: {metrics['threshold_metrics']['f1']:.4f}\n"
    report_str += f"Accuracy: {metrics['threshold_metrics']['accuracy']:.4f}\n"
    report_str += "Submission Note: Kaggle submission uses probabilities, not thresholded labels.\n\n"
    report_str += "Classification Report:\n"
    report_str += classification_report(y_true, y_pred_bin, digits=eval_config["report_digits"], zero_division=0)

    models_dir.mkdir(parents=True, exist_ok=True)
    model_artifact_path(models_dir, config, "evaluation_report").write_text(report_str, encoding="utf-8")
    with open(model_artifact_path(models_dir, config, "metrics"), "w", encoding="utf-8") as file:
        yaml.safe_dump(metrics, file, sort_keys=False)
    with open(model_artifact_path(models_dir, config, "threshold"), "w", encoding="utf-8") as file:
        yaml.safe_dump(threshold_info, file, sort_keys=False)
    if config["training"]["reports"]["save_threshold_table"]:
        threshold_table.to_csv(model_artifact_path(models_dir, config, "threshold_table"), index=False)
    if config["training"]["reports"]["save_lift_table"]:
        lift_table.to_csv(model_artifact_path(models_dir, config, "lift_table"), index=False)
    save_oof_predictions(y_true, y_pred_prob, ids, models_dir, config)
    save_diagnostic_plots(y_true, y_pred_prob, y_pred_bin, lift_table, model_name, models_dir, config, evaluation_scope)
    return threshold_info


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


def comparison_training_config(config):
    comparison_config = config["training"]["model_comparison"]
    resolved = deepcopy(config)
    resolved["training"]["cv_splits"] = comparison_config["cv_splits"]
    resolved["training"]["optuna_n_trials"] = comparison_config["max_trials_per_model"]
    resolved["training"]["optuna_subsample_rate"] = comparison_config["subsample_rate"]
    return resolved


def model_comparison_sample(X, y, ids, config):
    comparison_config = config["training"]["model_comparison"]
    subsample_rate = comparison_config["subsample_rate"]
    if subsample_rate >= 1.0:
        return X, y, ids
    X_sample, _, y_sample, _, ids_sample, _ = train_test_split(
        X,
        y,
        ids,
        train_size=subsample_rate,
        stratify=y,
        random_state=config["globals"]["random_state"],
    )
    return X_sample, y_sample, ids_sample


def compare_one_model(X, y, config, est_config, seed):
    t_config = config["training"]
    name = est_config["name"]
    search_space = est_config.get("search_space", {})
    feature_selection_enabled = t_config["preprocessing"]["feature_selection"]["enabled_during_search"]

    def objective(trial):
        trial_params = suggest_params(trial, search_space)
        model_params = {key.replace("model__", ""): value for key, value in trial_params.items()}
        candidate_config = merged_estimator_config(est_config, model_params)
        preds = cross_validated_single_predictions(
            X,
            y,
            candidate_config,
            config,
            desc=f"{name} Compare CV",
            is_trial=True,
            feature_selection_enabled=feature_selection_enabled,
        )
        return calculate_metric(y, preds, t_config["optimization_metric"], t_config["classification_threshold"])

    study = optuna.create_study(
        direction=t_config["optuna_direction"],
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=t_config["optuna_n_trials"], show_progress_bar=True)
    best_params = {key.replace("model__", ""): value for key, value in study.best_params.items()}
    best_config = merged_estimator_config(est_config, best_params)
    preds = cross_validated_single_predictions(
        X,
        y,
        best_config,
        config,
        desc=f"{name} Compare Best CV",
        is_trial=True,
        feature_selection_enabled=feature_selection_enabled,
    )
    return {
        "model": name,
        "evaluation_scope": "search_subsample_cv",
        "metric_name": t_config["optimization_metric"],
        "metric_value": float(calculate_metric(y, preds, t_config["optimization_metric"], t_config["classification_threshold"])),
        "roc_auc": float(roc_auc_score(y, preds)),
        "average_precision": float(average_precision_score(y, preds)),
        "brier_score": float(brier_score_loss(y, preds)),
        "trials": int(t_config["optuna_n_trials"]),
        "cv_splits": int(t_config["cv_splits"]),
        "subsample_rate": float(t_config["optuna_subsample_rate"]),
        "best_params": best_params,
    }


def run_model_comparison(X, y, ids, config, models_dir, seed, metadata):
    comparison_config = config["training"]["model_comparison"]
    if not comparison_config["enabled"]:
        return

    compare_config = comparison_training_config(config)
    X_compare, y_compare, _ = model_comparison_sample(X, y, ids, compare_config)
    rows = []
    for model_name in comparison_config["candidates"]:
        est_config = get_estimator_config_by_name(compare_config, model_name)
        rows.append(compare_one_model(X_compare, y_compare, compare_config, est_config, seed))

    csv_rows = [{key: value for key, value in row.items() if key != "best_params"} for row in rows]
    pd.DataFrame(csv_rows).sort_values("metric_value", ascending=False).to_csv(
        model_artifact_path(models_dir, config, "model_comparison_csv"),
        index=False,
    )
    payload = {
        "evaluation_scope": "search_subsample_cv",
        "metric_name": config["training"]["optimization_metric"],
        "rows": rows,
    }
    with open(model_artifact_path(models_dir, config, "model_comparison_yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    metadata["model_comparison"] = payload


def run_training(config):
    config = resolve_training_config(config)
    print(f"Initializing training pipeline ({config['training']['run_mode']})...")
    t_config = config["training"]
    seed = config["globals"]["random_state"]
    models_root = Path(t_config["artifact_paths"]["models_dir"])
    models_root.mkdir(parents=True, exist_ok=True)
    models_dir, experiment_id, timestamp = create_experiment_dir(models_root, config)
    save_config_snapshot(models_dir, config)
    print(f"Experiment artifacts: {models_dir}")
    configure_optuna_logging(config)

    train_path = Path(config["data"]["final"]["train"])
    test_path = Path(config["data"]["final"]["test"])
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}. Run --process first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}. Run --process first.")

    df = pd.read_csv(train_path)
    train_ids = df[t_config["id_col"]]
    y = df[t_config["target_col"]]
    X = clean_column_names(df.drop(columns=[t_config["target_col"], t_config["id_col"]]))
    metadata = build_run_metadata(config, X, y, train_path, experiment_id, timestamp)
    validate_reusable_artifacts(models_dir, config, metadata)

    run_model_comparison(X, y, train_ids, config, models_dir, seed, metadata)
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
        print("Skipping full-data OOF validation by run profile.")

    if phases["final_fit"]:
        model, preprocessor, accelerator = fit_final_single_model(X, y, best_config, config, models_dir, name)
        metadata["selected_accelerators"][name] = accelerator
        predict_test_and_submit(model, config, models_dir, preprocessor=preprocessor)


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


def predict_test_and_submit(model_obj, config, models_dir, preprocessor=None):
    print("Generating predictions...")
    if config["training"]["calibration"].get("apply_to_submission", False):
        raise ValueError("calibration.apply_to_submission is not supported in this diagnostics-only iteration.")

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
    print(f"Submission saved to {submission_path}")
