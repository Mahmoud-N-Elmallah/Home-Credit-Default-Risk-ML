import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.common.artifacts import model_artifact_path


def calculate_metric(y_true, y_pred_prob, metric_name, threshold):
    if metric_name == "roc_auc":
        return roc_auc_score(y_true, y_pred_prob)
    if metric_name == "average_precision":
        return average_precision_score(y_true, y_pred_prob)
    if metric_name == "f1":
        return f1_score(y_true, (y_pred_prob > threshold).astype(int))
    raise ValueError(f"Unknown metric: {metric_name}")


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
    eval_config = config["training"]["evaluation"]
    height = max(
        float(eval_config["feature_importance_min_height"]),
        min(
            float(eval_config["feature_importance_max_height"]),
            float(eval_config["feature_importance_height_per_feature"]) * len(top),
        ),
    )
    plt.figure(figsize=(float(eval_config["feature_importance_fig_width"]), height))
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


def save_diagnostic_plots(y_true, y_pred_prob, y_pred_bin, model_name, models_dir, config, evaluation_scope):
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
    plt.figure(figsize=tuple(eval_config["roc_curve_figsize"]))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.tight_layout()
    plt.savefig(model_artifact_path(models_dir, config, "roc_curve"))
    plt.close()


def save_evaluation_report(y_true, y_pred_prob, model_name, models_dir, config, evaluation_scope, ids=None):
    allowed_scopes = {"out_of_fold", "search_subsample_cv", "final_train_fit"}
    if evaluation_scope not in allowed_scopes:
        raise ValueError(f"Invalid evaluation scope: {evaluation_scope}")

    t_config = config["training"]
    eval_config = t_config["evaluation"]
    threshold_info, threshold_table = choose_threshold(y_true, y_pred_prob, config, evaluation_scope)
    threshold = threshold_info["threshold"]
    y_pred_bin = (y_pred_prob > threshold).astype(int)
    metrics = {
        "model": model_name,
        "evaluation_scope": evaluation_scope,
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
    report_str = f"Model: {model_name}\n"
    report_str += "=" * 40 + "\n"
    report_str += f"Evaluation Scope: {evaluation_scope}\n"
    report_str += f"Classification Threshold: {threshold:.4f}\n"
    report_str += f"Threshold Source: {threshold_info['source']}\n"
    report_str += f"Threshold Objective: {threshold_info['objective']}\n"
    report_str += f"ROC AUC Score: {metrics['ranking']['roc_auc']:.4f}\n"
    report_str += f"Average Precision (PR AUC): {metrics['ranking']['average_precision']:.4f}\n"
    report_str += f"Brier Score: {metrics['ranking']['brier_score']:.4f}\n"
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
    save_diagnostic_plots(y_true, y_pred_prob, y_pred_bin, model_name, models_dir, config, evaluation_scope)
    return threshold_info
