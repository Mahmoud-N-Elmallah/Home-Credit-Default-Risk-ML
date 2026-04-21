import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from catboost import Pool
from sklearn.model_selection import train_test_split

from src.model_training.run_training import clean_column_names


DEFAULT_EXPERIMENT_DIR = "Models/20260420_174015_balanced_catboost"


def parse_args():
    parser = argparse.ArgumentParser(description="Run native CatBoost SHAP analysis for a trained experiment.")
    parser.add_argument("--config", default="config.yaml", help="Path to project config YAML.")
    parser.add_argument("--experiment-dir", default=DEFAULT_EXPERIMENT_DIR, help="Trained experiment directory.")
    parser.add_argument("--sample-size", type=int, default=5000, help="Stratified train sample size for SHAP.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top SHAP features to plot.")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def sample_training_rows(df, target_col, sample_size, seed):
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    if sample_size >= len(df):
        return df.copy()
    sample, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[target_col],
        random_state=seed,
    )
    return sample.sort_index().copy()


def expected_preprocessor_input_columns(preprocessor):
    if getattr(preprocessor, "scaler", None) is not None and hasattr(preprocessor.scaler, "feature_names_in_"):
        return preprocessor.scaler.feature_names_in_.tolist()
    if getattr(preprocessor, "selector", None) is not None and hasattr(preprocessor.selector, "feature_names_in_"):
        return preprocessor.selector.feature_names_in_.tolist()
    return None


def align_to_preprocessor_input(X, preprocessor):
    expected_columns = expected_preprocessor_input_columns(preprocessor)
    if expected_columns is None:
        return X, {"aligned": False, "missing_input_columns": [], "extra_input_columns": []}

    current_columns = set(X.columns)
    expected_set = set(expected_columns)
    missing_columns = [col for col in expected_columns if col not in current_columns]
    extra_columns = [col for col in X.columns if col not in expected_set]
    X_aligned = X.reindex(columns=expected_columns, fill_value=0)
    return X_aligned, {
        "aligned": bool(missing_columns or extra_columns),
        "missing_input_columns": missing_columns,
        "extra_input_columns": extra_columns,
    }


def load_and_transform_sample(config, experiment_dir, sample_size):
    artifact_paths = config["training"]["artifact_paths"]
    model_path = experiment_dir / artifact_paths["single_model"]
    preprocessor_path = experiment_dir / artifact_paths["preprocessor"]
    train_path = resolve_path(config["data"]["final"]["train"])
    target_col = config["training"]["target_col"]
    id_col = config["training"]["id_col"]
    seed = config["globals"]["random_state"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Final training data not found: {train_path}. Run --process first.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    if not hasattr(preprocessor, "pruned_columns"):
        preprocessor.pruned_columns = None
    df = pd.read_csv(train_path)
    sample = sample_training_rows(df, target_col, sample_size, seed)

    ids = sample[id_col].reset_index(drop=True)
    y = sample[target_col].reset_index(drop=True)
    X = clean_column_names(sample.drop(columns=[target_col, id_col])).reset_index(drop=True)
    X, alignment = align_to_preprocessor_input(X, preprocessor)
    X_processed = preprocessor.transform(X)
    probabilities = model.predict_proba(X_processed)[:, 1]

    native_importance = model.get_feature_importance()
    if X_processed.shape[1] != len(native_importance):
        raise ValueError(
            "Transformed feature count does not match model feature count: "
            f"{X_processed.shape[1]} vs {len(native_importance)}"
        )

    return {
        "model": model,
        "X_processed": X_processed,
        "ids": ids,
        "y": y,
        "probabilities": probabilities,
        "native_importance": native_importance,
        "alignment": alignment,
        "paths": {
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
            "train_path": train_path,
        },
    }


def compute_shap_values(model, X_processed):
    pool = Pool(X_processed, feature_names=X_processed.columns.to_list())
    shap_output = model.get_feature_importance(data=pool, type="ShapValues")
    shap_values = shap_output[:, :-1]
    expected_values = shap_output[:, -1]
    return shap_values, expected_values


def build_feature_importance(feature_names, shap_values, native_importance):
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
            "std_shap": shap_values.std(axis=0),
            "native_importance": native_importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def build_sample_predictions(ids, y, probabilities, expected_values, feature_names, shap_values):
    top_positive_idx = np.argmax(shap_values, axis=1)
    top_negative_idx = np.argmin(shap_values, axis=1)
    return pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": y,
            "predicted_probability": probabilities,
            "expected_value": expected_values,
            "top_positive_feature": [feature_names[idx] for idx in top_positive_idx],
            "top_positive_shap": shap_values[np.arange(len(shap_values)), top_positive_idx],
            "top_negative_feature": [feature_names[idx] for idx in top_negative_idx],
            "top_negative_shap": shap_values[np.arange(len(shap_values)), top_negative_idx],
        }
    )


def plot_summary_bar(feature_importance, plots_dir, top_n):
    top = feature_importance.head(top_n).sort_values("mean_abs_shap", ascending=True)
    height = max(6, len(top) * 0.28)
    plt.figure(figsize=(10, height))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#4c78a8")
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {len(top)} SHAP Feature Contributions")
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_summary_bar.png", dpi=160)
    plt.close()


def plot_beeswarm_like(X_processed, shap_values, feature_importance, plots_dir, top_n, seed):
    top_features = feature_importance.head(top_n)["feature"].to_list()
    rng = np.random.default_rng(seed)
    plt.figure(figsize=(11, max(7, len(top_features) * 0.3)))

    for y_pos, feature in enumerate(reversed(top_features)):
        col_idx = X_processed.columns.get_loc(feature)
        values = X_processed[feature].to_numpy()
        shap_col = shap_values[:, col_idx]
        jitter = rng.uniform(-0.28, 0.28, size=len(shap_col))
        low, high = np.nanpercentile(values, [1, 99])
        if np.isclose(low, high):
            colors = np.zeros_like(values, dtype=float)
        else:
            colors = np.clip((values - low) / (high - low), 0, 1)
        plt.scatter(shap_col, y_pos + jitter, c=colors, cmap="coolwarm", s=8, alpha=0.55, linewidths=0)

    plt.axvline(0, color="#333333", linewidth=1)
    plt.yticks(range(len(top_features)), list(reversed(top_features)))
    plt.xlabel("SHAP value")
    plt.title(f"SHAP Beeswarm-Style Plot: Top {len(top_features)} Features")
    cbar = plt.colorbar()
    cbar.set_label("Scaled feature value")
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_beeswarm_top.png", dpi=160)
    plt.close()


def plot_dependence(X_processed, shap_values, feature_importance, plots_dir, count=3):
    for rank, feature in enumerate(feature_importance.head(count)["feature"], start=1):
        col_idx = X_processed.columns.get_loc(feature)
        plt.figure(figsize=(7, 5))
        plt.scatter(X_processed[feature], shap_values[:, col_idx], s=8, alpha=0.4, color="#4c78a8")
        plt.axhline(0, color="#333333", linewidth=1)
        plt.xlabel(f"{feature} (processed value)")
        plt.ylabel("SHAP value")
        plt.title(f"SHAP Dependence: {feature}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"shap_dependence_top_{rank}.png", dpi=160)
        plt.close()


def save_metadata(config, experiment_dir, paths, alignment, sample_size, actual_sample_size, top_n, reports_dir):
    metadata = {
        "experiment_dir": str(experiment_dir),
        "sample_size": int(sample_size),
        "actual_sample_size": int(actual_sample_size),
        "top_n": int(top_n),
        "model_path": str(paths["model_path"]),
        "preprocessor_path": str(paths["preprocessor_path"]),
        "data_path": str(paths["train_path"]),
        "random_state": int(config["globals"]["random_state"]),
        "input_alignment": {
            "aligned": bool(alignment["aligned"]),
            "missing_input_column_count": len(alignment["missing_input_columns"]),
            "extra_input_column_count": len(alignment["extra_input_columns"]),
            "missing_input_columns_preview": alignment["missing_input_columns"][:20],
            "extra_input_columns_preview": alignment["extra_input_columns"][:20],
        },
    }
    with open(reports_dir / "shap_metadata.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)


def run_shap_analysis(config_path, experiment_dir, sample_size, top_n):
    config = load_yaml(config_path)
    reports_dir = experiment_dir / "reports"
    plots_dir = experiment_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    payload = load_and_transform_sample(config, experiment_dir, sample_size)
    X_processed = payload["X_processed"]
    shap_values, expected_values = compute_shap_values(payload["model"], X_processed)
    feature_names = X_processed.columns.to_list()
    feature_importance = build_feature_importance(feature_names, shap_values, payload["native_importance"])
    sample_predictions = build_sample_predictions(
        payload["ids"],
        payload["y"],
        payload["probabilities"],
        expected_values,
        feature_names,
        shap_values,
    )

    feature_importance.to_csv(reports_dir / "shap_feature_importance.csv", index=False)
    sample_predictions.to_csv(reports_dir / "shap_sample_predictions.csv", index=False)
    save_metadata(
        config,
        experiment_dir,
        payload["paths"],
        payload["alignment"],
        sample_size,
        len(X_processed),
        top_n,
        reports_dir,
    )

    plot_summary_bar(feature_importance, plots_dir, top_n)
    plot_beeswarm_like(
        X_processed,
        shap_values,
        feature_importance,
        plots_dir,
        top_n,
        config["globals"]["random_state"],
    )
    plot_dependence(X_processed, shap_values, feature_importance, plots_dir)

    print(f"SHAP analysis saved to {experiment_dir}")
    print(feature_importance.head(min(top_n, 10)).to_string(index=False))


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    experiment_dir = resolve_path(args.experiment_dir)
    run_shap_analysis(config_path, experiment_dir, args.sample_size, args.top_n)


if __name__ == "__main__":
    main()
