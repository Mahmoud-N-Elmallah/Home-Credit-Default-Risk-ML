import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.model_training.models import get_accelerator_params, resolve_model_accelerator


SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


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
