# Home Credit Default Risk ML

Final project for the ITI Machine Learning course.

This project builds a Home Credit default-risk model from the Kaggle raw CSV tables. The data pipeline creates customer-level train/test feature files with target-free application, bureau, previous-loan, POS, installment, and credit-card features. The training pipeline tunes/evaluates one configured primary model with out-of-fold validation before writing a Kaggle submission file.

## Setup

```powershell
uv sync
```

Use the project virtual environment when running commands:

```powershell
.\.venv\Scripts\python.exe main.py --help
```

## Data Layout

Place the Kaggle CSV files under `Data/Raw/`:

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `POS_CASH_balance.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`
- `sample_submission.csv`
- `HomeCredit_columns_description.csv`

## Commands

Process raw data:

```powershell
.\.venv\Scripts\python.exe main.py --process
```

Train model and generate submission:

```powershell
.\.venv\Scripts\python.exe main.py --train
```

Training defaults to the `balanced` profile. Change `training.run_mode` in `config.yaml` to:

- `fast_dev` for quick smoke runs.
- `balanced` for default speed/quality.
- `max_quality` for slower, broader validation/search.

Run both:

```powershell
.\.venv\Scripts\python.exe main.py --all
```

## Outputs

- `Data/final/final_train.csv`
- `Data/final/final_test.csv`
- `Data/final/feature_manifest.yaml`
- `Models/latest_experiment.txt`
- `Models/experiments/<experiment_id>/submission.csv`
- `Models/experiments/<experiment_id>/final_model.pkl`
- `Models/experiments/<experiment_id>/training_preprocessor.pkl`
- `Models/experiments/<experiment_id>/training_run_metadata.yaml`
- `Models/experiments/<experiment_id>/threshold.yaml`
- `Models/experiments/<experiment_id>/reports/*`
- `Models/experiments/<experiment_id>/plots/*`

## Configuration

Project behavior is controlled from `config.yaml`: file paths, CSV parsing, feature engineering sets, feature source columns, recency windows, last-N windows, aggregation source columns/prefixes, cleanup thresholds, categorical encoding, feature selection, imbalance handling, acceleration fallback, run profiles, training phases, experiment folders, threshold tuning, primary model choice, candidate model parameters, Optuna search spaces, artifact names, and evaluation settings.

Training is single-model by design. Set `training.models.primary` to one of `training.models.candidates` to choose CatBoost, LightGBM, or XGBoost without changing code.

Training uses `training.acceleration.preferred` first and retries `training.acceleration.fallback` when GPU support fails. Model-specific GPU/CPU parameters live under `training.acceleration.models`.

Training phases are explicit: `search`, `validate`, and `final_fit`. Reports label metric scope as `search_subsample_cv`, `out_of_fold`, or `final_train_fit`. Classification reports use an OOF-tuned threshold; Kaggle submissions remain probabilities.
