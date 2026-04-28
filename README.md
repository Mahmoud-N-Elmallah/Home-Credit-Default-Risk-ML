# Home Credit Default Risk ML

Machine-learning pipeline for [**Home Credit Default Risk**](https://www.kaggle.com/c/home-credit-default-risk).
It builds customer-level features from raw Home Credit tables, trains a configured model, evaluates with out-of-fold validation, and writes a Kaggle submission CSV.

Results: **0.792 public leaderboard ROC AUC** .

## Pipeline

![Project pipeline diagram](docs/project_pipeline_diagram.svg)

1. **Download data**: ensure the Kaggle competition CSVs exist under
   `Data/Raw/`.
2. **Process data**: read raw Kaggle CSVs, build application/bureau/history
   features, align train/test, and write `Data/final/`.
3. **Train model**: tune the primary model, validate with OOF metrics, tune
   report thresholds, fit the final model, and write artifacts under
   `Models/<experiment_id>/`.

Default primary model is **CatBoost**. LightGBM and XGBoost remain configured
as available candidates.

## Setup

```powershell
uv sync
uv run python main.py --help
```

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

## Data

Raw data comes from the Kaggle
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
competition. Configure Kaggle credentials with `~/.kaggle/kaggle.json` or
`KAGGLE_USERNAME` / `KAGGLE_KEY`, and accept the competition rules on Kaggle.

Then run:

```powershell
uv run python main.py run.step=download
```

This no-ops when all expected raw files already exist under `Data/Raw/`:

```text
application_train.csv
application_test.csv
bureau.csv
bureau_balance.csv
previous_application.csv
POS_CASH_balance.csv
installments_payments.csv
credit_card_balance.csv
sample_submission.csv
HomeCredit_columns_description.csv
```

`Data/` is git-ignored.

Large data is tracked with DVC, not Git. After configuring the DVC remote:

```powershell
uv run dvc pull
uv run dvc push
```

The reproducible process/train pipeline is defined in `dvc.yaml`:

```powershell
uv run dvc repro
```

The DVC order is `download -> process -> train`. DVC training disables MLflow
remote logging by default so `dvc repro` stays a local reproducibility command;
use `uv run python main.py run.step=train` when you want MLflow/DagsHub tracking.

Data validation runs during `process` and writes
`Data/final/validation_report.yaml`.

## Run

```powershell
# download raw data if missing
uv run python main.py run.step=download

# process raw data
uv run python main.py run.step=process

# train and generate submission
uv run python main.py run.step=train

# run download, process, and train
uv run python main.py run.step=all
```

Hydra overrides are supported:

```powershell
uv run python main.py run.step=train training.cv_splits=2 training.optuna_n_trials=5
```

If preprocessing config changes, run `run.step=process` before training so
`Data/final/` matches the current pipeline.

## Training Controls

Main runtime controls live in `conf/config.yaml`: model choice, training budget,
preprocessing strategy, MLflow settings, DVC experiment name, and data paths.
Fixed artifact names, plot styling, and GPU fallback policy live in code.

Faster smoke run:

```powershell
uv run python main.py run.step=train `
  training.cv_splits=2 `
  training.optuna_n_trials=5 `
  training.optuna_subsample_rate=0.15 `
  training.run_full_oof_validation=false
```

Hydra multirun can populate several MLflow runs:

```powershell
uv run python main.py -m run.step=train `
  training.models.primary=catboost `
  training.preprocessing.imbalance.strategy=smote,borderline_smote,adasyn `
  training.preprocessing.scaler=standard,robust,minmax `
  training.optuna_n_trials=5 `
  training.optuna_subsample_rate=0.15 `
  training.cv_splits=2
```

## Inference

Score already processed feature rows:

```powershell
uv run python src\inference.py --input Data\final\final_test.csv
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
```

Each experiment keeps its own artifacts: model, preprocessor, submission,
metrics, report, feature importance, ROC/confusion plots, config snapshot, and
logs. DVC owns the reproducible `Models/dvc_train` output; MLflow mirrors
curated experiment metadata and artifacts for comparison.

Submission file is `Models/<experiment_id>/submission.csv` with
`SK_ID_CURR,TARGET`; `TARGET` is a probability for Kaggle ROC AUC scoring.

## Experiment Tracking

MLflow is configured for DagsHub by default. Set DagsHub auth before remote
tracking, or disable tracking for local-only runs with
`tracking.mlflow.enabled=false`.

Training logs params, metrics, curated artifacts, and a PyFunc model wrapper to
MLflow. Accepted runs are registered as `home-credit-default-risk` and assigned
the `champion` alias when the configured metric gate passes.

## Config

`conf/config.yaml` is the Hydra source of truth for values expected to change
between runs. Fixed Kaggle schema, artifact filenames, plot styling, report
formatting, and accelerator fallback behavior live in code constants.

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

## Model Visuals

### ROC Curve Comparison

| Best Balanced CatBoost | Vanilla Baseline CatBoost |
| --- | --- |
| ![ROC curve for the best CatBoost experiment](Models/20260420_174015_balanced_catboost/plots/roc_curve.png) | ![ROC curve for the vanilla baseline data with CatBoost experiment](Models/baseline/roc_curve.png) |

### Feature Importance

![Top feature importances for the best CatBoost experiment](Models/20260420_174015_balanced_catboost/plots/feature_importance_top.png)

### SHAP Analysis

| Mean Absolute SHAP Contributions | SHAP Beeswarm-Style View |
| --- | --- |
| ![SHAP summary bar plot for the best CatBoost experiment](Models/20260420_174015_balanced_catboost/plots/shap_summary_bar.png) | ![SHAP beeswarm-style plot for the best CatBoost experiment](Models/20260420_174015_balanced_catboost/plots/shap_beeswarm_top.png) |
