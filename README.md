# Home Credit Default Risk ML

Leakage-safe machine-learning pipeline for Kaggle
[**Home Credit Default Risk**](https://www.kaggle.com/c/home-credit-default-risk).
It builds customer-level features from the raw Home Credit tables, trains one
configured primary model, evaluates with out-of-fold validation, and writes a
Kaggle submission.

Best project result: **0.79074 public leaderboard ROC AUC** from
`Models/20260420_174015_balanced_catboost`.

## Pipeline

![Project pipeline diagram](docs/project_pipeline_diagram.svg)

1. **Process data**: read raw Kaggle CSVs, build application/bureau/history
   features, align train/test, and write `Data/final/`.
2. **Train model**: tune the primary model, validate with OOF predictions, tune
   report thresholds, fit the final model, and write artifacts under
   `Models/<experiment_id>/`.

Default primary model is **CatBoost**. LightGBM and XGBoost remain configured
as available candidates.

## Setup

```powershell
uv sync
.\.venv\Scripts\python.exe main.py --help
```

Use Python `3.13`. Dependencies are declared in `pyproject.toml` and locked in
`uv.lock`.

## Data

Place Kaggle raw files under `Data/Raw/`:

```text
application_train.csv
application_test.csv
bureau.csv
bureau_balance.csv
previous_application.csv
POS_CASH_balance.csv
installments_payments.csv
credit_card_balance.csv
```

`Data/` is git-ignored.

## Run

```powershell
# process raw data
.\.venv\Scripts\python.exe main.py run.step=process

# train and generate submission
.\.venv\Scripts\python.exe main.py run.step=train

# run both stages
.\.venv\Scripts\python.exe main.py run.step=all
```

Hydra overrides are supported:

```powershell
.\.venv\Scripts\python.exe main.py run.step=train training.cv_splits=2 training.optuna_n_trials=5
```

If preprocessing config changes, run `run.step=process` before training so
`Data/final/` matches the current pipeline.

## Training Controls

Main runtime controls in `conf/config.yaml`:

```yaml
training:
  cv_splits: 3
  optuna_n_trials: 10
  optuna_subsample_rate: 0.35
  run_full_oof_validation: true
  models:
    primary: "catboost"
```

Faster smoke run:

```powershell
.\.venv\Scripts\python.exe main.py run.step=train training.cv_splits=2 training.optuna_n_trials=5 training.optuna_subsample_rate=0.15 training.run_full_oof_validation=false
```

## Inference

Score already processed feature rows:

```powershell
.\.venv\Scripts\python.exe src\inference.py --input Data\final\final_test.csv
```

The input must match the processed feature schema. Raw application rows alone
are not enough because the model uses engineered aggregate features.

## Outputs

Processing writes:

```text
Data/final/final_train.csv
Data/final/final_test.csv
Data/final/feature_manifest.yaml
```

Training writes:

```text
Models/<experiment_id>/
Models/latest_experiment.txt
```

Common artifacts:

```text
config_snapshot.yaml
training_run_metadata.yaml
logs/training.log
final_model.pkl
training_preprocessor.pkl
threshold.yaml
submission.csv
reports/
plots/
```

Submission file: `Models/<experiment_id>/submission.csv`, with
`SK_ID_CURR,TARGET`; `TARGET` is a probability for Kaggle ROC AUC scoring.

## Config

`conf/config.yaml` is the Hydra source of truth for runtime paths, training
controls, model settings, artifact names, and analysis/inference defaults.

Fixed Kaggle schema, source column names, engineered feature names, and
aggregation specs live in `src/data_processing/constants.py`. Historical
baseline notebook settings live in `Notebooks/baseline_config.yaml`.

## Source Layout

```text
src/
  common/           shared artifact, config, logging, and schema helpers
  data_processing/  raw CSV loading, constants, encoding, aggregations, features
  model_training/   training artifacts, preprocessing, models, search, evaluation
  inference/        reusable batch scoring core and CLI parser
  analysis/         SHAP analysis workflow
```

Compatibility wrappers remain:

```text
src/inference.py
src/shap_analysis.py
```

## Model Visuals

### ROC Curve Comparison

| Best Balanced CatBoost | Vanilla Baseline CatBoost |
| --- | --- |
| ![ROC curve for the best balanced CatBoost experiment](Models/20260420_174015_balanced_catboost/plots/roc_curve.png) | ![ROC curve for the vanilla baseline CatBoost experiment](Models/baseline/roc_curve.png) |

### Feature Importance

![Top feature importances for the best balanced CatBoost experiment](Models/20260420_174015_balanced_catboost/plots/feature_importance_top.png)

### SHAP Analysis

| Mean Absolute SHAP Contributions | SHAP Beeswarm-Style View |
| --- | --- |
| ![SHAP summary bar plot for the best balanced CatBoost experiment](Models/20260420_174015_balanced_catboost/plots/shap_summary_bar.png) | ![SHAP beeswarm-style plot for the best balanced CatBoost experiment](Models/20260420_174015_balanced_catboost/plots/shap_beeswarm_top.png) |
