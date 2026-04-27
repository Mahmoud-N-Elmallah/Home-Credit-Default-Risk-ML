# Home Credit Default Risk ML

Lemachine-learning pipeline for [**Home Credit Default Risk**](https://www.kaggle.com/c/home-credit-default-risk).
It builds customer-level features from the raw Home Credit tables,trains configured model, evaluates with out-of-fold validation, and writes a Kaggle submission csv.

Results: **0.79074 public leaderboard ROC AUC** .

## Pipeline

![Project pipeline diagram](docs/project_pipeline_diagram.svg)

1. **Process data**: read raw Kaggle CSVs, build application/bureau/history
   features, align train/test, and write `Data/final/`.
2. **Train model**: tune the primary model, validate with OOF metrics, tune
   report thresholds, fit the final model, and write artifacts under
   `Models/<experiment_id>/`.

Default primary model is **CatBoost** as choosen from mlflow tracking of the experiments. LightGBM and XGBoost remain configured
as available candidates.

## Setup

```powershell
uv sync
uv run python main.py --help
```

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

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

Large data is tracked with DVC, not Git. After configuring the DVC remote:

```powershell
uv run dvc pull
uv run dvc push
```

## Run

```powershell
# process raw data
uv run python main.py run.step=process

# train and generate submission
uv run python main.py run.step=train

# run both stages
uv run python main.py run.step=all
```

Hydra overrides are supported:

```powershell
uv run python main.py run.step=train training.cv_splits=2 training.optuna_n_trials=5
```

If preprocessing config changes, run `run.step=process` before training so
`Data/final/` matches the current pipeline.

## Training Controls

Main runtime controls live in `conf/config.yaml`. 

Faster smoke run:

```powershell
uv run python main.py run.step=train `
  training.cv_splits=2 `
  training.optuna_n_trials=5 `
  training.optuna_subsample_rate=0.15 `
  training.run_full_oof_validation=false
```

Hydra multirun populate several MLflow runs:

```powershell
uv run python main.py -m run.step=train `
  training.models.primary=catboost,lightgbm,xgboost `
  training.preprocessing.imbalance.strategy=smote,borderline_smote,adasyn,undersample,oversample `
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

Each experiment keeps its own artifacts and are being tracked using mlflow and dvc such as the model, preprocessor, submission, metrics, report,
feature importance, ROC/confusion plots, config snapshot, and logs.

Submission file is `Models/<experiment_id>/submission.csv` with
`SK_ID_CURR,TARGET`; `TARGET` is a probability for Kaggle ROC AUC scoring.

## Experiment Tracking

MLflow is configured for DagsHub by default. Local artifacts remain the source
of truth; MLflow mirrors core params, metrics, tags, and curated artifacts.
The option to disable tracking for local-only runs is avalaible using hydra config override with `tracking.mlflow.enabled=false`.

## Config

`conf/config.yaml` is the Hydra source of truth for runtime paths, training
controls, tracking settings, model settings, artifact names, and
analysis/inference defaults.

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
