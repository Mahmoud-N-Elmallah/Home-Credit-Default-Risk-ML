# Home Credit Default Risk ML

Final project for the ITI Machine Learning course.

This project builds a Home Credit default-risk model from the Kaggle raw CSV tables. The data pipeline creates customer-level train/test feature files, and the training pipeline tunes/evaluates models with out-of-fold validation before writing a Kaggle submission file.

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
- `Data/final/submission.csv`
- `Models/*_evaluation_report.txt`
- `Models/*_confusion_matrix.png`
- `Models/training_preprocessor.pkl`
- `Models/training_run_metadata.yaml`
- `Models/*_best_model.pkl`

## Configuration

Project behavior is controlled from `config.yaml`: file paths, CSV parsing, feature engineering thresholds, aggregation source columns/prefixes, categorical encoding, feature selection, imbalance handling, acceleration fallback, run profiles, training phases, model parameters, Optuna search spaces, artifact names, and evaluation settings.

Training uses `training.acceleration.preferred` first and retries `training.acceleration.fallback` when GPU support fails. Model-specific GPU/CPU parameters live under `training.acceleration.models`.

Training phases are explicit: `search`, `validate`, and `final_fit`. Reports label metric scope as `search_subsample_cv`, `out_of_fold`, or `final_train_fit`.
