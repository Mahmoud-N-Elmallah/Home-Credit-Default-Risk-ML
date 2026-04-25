import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.model_selection import train_test_split

from src.common.config_io import load_yaml, resolve_project_path
from src.common.schema import clean_column_names, expected_preprocessor_input_columns

resolve_path = resolve_project_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run native CatBoost SHAP analysis for a trained experiment.")
    parser.add_argument("--config", default="config.yaml", help="Path to project config YAML.")
    parser.add_argument("--experiment-dir", help="Trained experiment directory. Defaults to analysis.shap config.")
    parser.add_argument("--sample-size", type=int, help="Stratified train sample size for SHAP.")
    parser.add_argument("--top-n", type=int, help="Number of top SHAP features to plot.")
    return parser.parse_args()


def shap_config(config):
    return config["analysis"]["shap"]


def output_path(experiment_dir, relative_path):
    path = Path(relative_path)
    resolved = path if path.is_absolute() else experiment_dir / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


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


def plot_summary_bar(feature_importance, output_file, top_n, plot_config):
    top = feature_importance.head(top_n).sort_values("mean_abs_shap", ascending=True)
    height = max(
        float(plot_config["bar_fig_min_height"]),
        len(top) * float(plot_config["bar_height_per_feature"]),
    )
    plt.figure(figsize=(float(plot_config["bar_fig_width"]), height))
    plt.barh(top["feature"], top["mean_abs_shap"], color=plot_config["bar_color"])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {len(top)} SHAP Feature Contributions")
    plt.tight_layout()
    plt.savefig(output_file, dpi=int(plot_config["dpi"]))
    plt.close()


def plot_beeswarm_like(X_processed, shap_values, feature_importance, output_file, top_n, seed, plot_config):
    top_features = feature_importance.head(top_n)["feature"].to_list()
    rng = np.random.default_rng(seed)
    plt.figure(
        figsize=(
            float(plot_config["beeswarm_fig_width"]),
            max(
                float(plot_config["beeswarm_fig_min_height"]),
                len(top_features) * float(plot_config["beeswarm_height_per_feature"]),
            ),
        )
    )

    for y_pos, feature in enumerate(reversed(top_features)):
        col_idx = X_processed.columns.get_loc(feature)
        values = X_processed[feature].to_numpy()
        shap_col = shap_values[:, col_idx]
        jitter_limit = float(plot_config["beeswarm_jitter"])
        jitter = rng.uniform(-jitter_limit, jitter_limit, size=len(shap_col))
        low, high = np.nanpercentile(values, plot_config["beeswarm_value_percentiles"])
        if np.isclose(low, high):
            colors = np.zeros_like(values, dtype=float)
        else:
            colors = np.clip((values - low) / (high - low), 0, 1)
        plt.scatter(
            shap_col,
            y_pos + jitter,
            c=colors,
            cmap=plot_config["beeswarm_cmap"],
            s=float(plot_config["beeswarm_marker_size"]),
            alpha=float(plot_config["beeswarm_alpha"]),
            linewidths=0,
        )

    plt.axvline(
        0,
        color=plot_config["zero_line_color"],
        linewidth=float(plot_config["zero_line_width"]),
    )
    plt.yticks(range(len(top_features)), list(reversed(top_features)))
    plt.xlabel("SHAP value")
    plt.title(f"SHAP Beeswarm-Style Plot: Top {len(top_features)} Features")
    cbar = plt.colorbar()
    cbar.set_label("Scaled feature value")
    plt.tight_layout()
    plt.savefig(output_file, dpi=int(plot_config["dpi"]))
    plt.close()


def plot_dependence(X_processed, shap_values, feature_importance, experiment_dir, shap_settings):
    plot_config = shap_settings["plot"]
    dependence_template = shap_settings["plots"]["dependence_template"]
    count = int(shap_settings["dependence_plot_count"])
    for rank, feature in enumerate(feature_importance.head(count)["feature"], start=1):
        col_idx = X_processed.columns.get_loc(feature)
        plt.figure(figsize=tuple(plot_config["dependence_figsize"]))
        plt.scatter(
            X_processed[feature],
            shap_values[:, col_idx],
            s=float(plot_config["dependence_marker_size"]),
            alpha=float(plot_config["dependence_alpha"]),
            color=plot_config["dependence_color"],
        )
        plt.axhline(
            0,
            color=plot_config["zero_line_color"],
            linewidth=float(plot_config["zero_line_width"]),
        )
        plt.xlabel(f"{feature} (processed value)")
        plt.ylabel("SHAP value")
        plt.title(f"SHAP Dependence: {feature}")
        plt.tight_layout()
        plt.savefig(
            output_path(experiment_dir, dependence_template.format(rank=rank)),
            dpi=int(plot_config["dpi"]),
        )
        plt.close()


def save_metadata(
    config,
    experiment_dir,
    paths,
    alignment,
    sample_size,
    actual_sample_size,
    top_n,
    metadata_path,
):
    preview_limit = int(shap_config(config)["metadata_preview_limit"])
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
            "missing_input_columns_preview": alignment["missing_input_columns"][:preview_limit],
            "extra_input_columns_preview": alignment["extra_input_columns"][:preview_limit],
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)


def run_shap_analysis(config_path, experiment_dir_arg=None, sample_size_arg=None, top_n_arg=None):
    config = load_yaml(config_path)
    shap_settings = shap_config(config)
    experiment_dir = resolve_path(experiment_dir_arg or config["artifacts"]["best_experiment_dir"])
    sample_size = int(sample_size_arg if sample_size_arg is not None else shap_settings["sample_size"])
    top_n = int(top_n_arg if top_n_arg is not None else shap_settings["top_n"])

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

    reports_config = shap_settings["reports"]
    plots_config = shap_settings["plots"]
    plot_config = shap_settings["plot"]

    feature_importance.to_csv(output_path(experiment_dir, reports_config["feature_importance"]), index=False)
    sample_predictions.to_csv(output_path(experiment_dir, reports_config["sample_predictions"]), index=False)
    save_metadata(
        config,
        experiment_dir,
        payload["paths"],
        payload["alignment"],
        sample_size,
        len(X_processed),
        top_n,
        output_path(experiment_dir, reports_config["metadata"]),
    )

    plot_summary_bar(
        feature_importance,
        output_path(experiment_dir, plots_config["summary_bar"]),
        top_n,
        plot_config,
    )
    plot_beeswarm_like(
        X_processed,
        shap_values,
        feature_importance,
        output_path(experiment_dir, plots_config["beeswarm"]),
        top_n,
        config["globals"]["random_state"],
        plot_config,
    )
    plot_dependence(X_processed, shap_values, feature_importance, experiment_dir, shap_settings)

    print(f"SHAP analysis saved to {experiment_dir}")
    preview_count = min(top_n, int(shap_settings["console_preview_limit"]))
    print(feature_importance.head(preview_count).to_string(index=False))


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    run_shap_analysis(config_path, args.experiment_dir, args.sample_size, args.top_n)
