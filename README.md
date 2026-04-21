# Home Credit Default Risk ML

Leakage-safe machine-learning pipeline for the Kaggle **Home Credit Default Risk**
competition. The project builds customer-level features from the raw Home Credit
tables, trains one configured primary model, evaluates it with out-of-fold
validation and produces a submission csv.

Best project result: **0.79074 public leaderboard ROC AUC** from
`Models/20260420_174015_balanced_catboost`.

## Project Pipeline

![Project pipeline diagram](docs/project_pipeline_diagram.svg)

## Model Visuals

### ROC Curve Comparison

| Best Balanced CatBoost | Vanilla Baseline CatBoost |
| --- | --- |
| ![ROC curve for the best balanced CatBoost experiment](Models/20260420_174015_balanced_catboost/plots/roc_curve.png) | ![ROC curve for the vanilla baseline CatBoost experiment](Models/baseline/roc_curve.png) |

### Classification Report Comparison

| Best Balanced CatBoost | Vanilla Baseline CatBoost |
| --- | --- |
| <pre>precision    recall  f1-score   support<br><br>0     0.9480    0.9009    0.9239    282686<br>1     0.2795    0.4377    0.3412     24825<br><br>accuracy                         0.8635    307511<br>macro avg     0.6138    0.6693    0.6325    307511<br>weighted avg  0.8941    0.8635    0.8768    307511</pre> | <pre>precision    recall  f1-score   support<br><br>0       0.96      0.72      0.82     56538<br>1       0.17      0.67      0.28      4965<br><br>accuracy                           0.72     61503<br>macro avg       0.57      0.70      0.55     61503<br>weighted avg    0.90      0.72      0.78     61503</pre> |

### Feature Importance

![Top feature importances for the best balanced CatBoost experiment](Models/20260420_174015_balanced_catboost/plots/feature_importance_top.png)

## What This Project Does

The pipeline has two main stages:

1. **Processing**
   - Reads the Kaggle raw CSV tables from `Data/Raw/`.
   - Builds application, bureau, previous application, POS cash, installment,
     and credit-card features.
   - Adds ratio features, recency-window features, last-N history features,
     missing-value indicators, and categorical encodings.
   - Aligns train/test columns and saves final modeling files under `Data/final/`.

2. **Training**
   - Resolves the selected run profile from `config.yaml`.
   - Tunes the primary model with Optuna.
   - Runs out-of-fold validation when the selected profile enables it.
   - Tunes the classification threshold from OOF/search-CV predictions.
   - Fits the final model on all training rows.
   - Saves metrics, plots, feature importance, model artifacts, and submission
     under `Models/<experiment_id>/`.

The default model is single-model **CatBoost**. LightGBM and XGBoost remain
available as configured candidates, but no ensemble path is used.

## Setup

This project uses Python `3.13` and `uv`.

```powershell
uv sync
```

Run commands with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe main.py --help
```

Main dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.
They include Polars, pandas, scikit-learn, imbalanced-learn, CatBoost, LightGBM,
XGBoost, Optuna, matplotlib, seaborn, PyYAML, and joblib.

## Data Layout

Place the Kaggle raw files here:

```text
Data/Raw/application_train.csv
Data/Raw/application_test.csv
Data/Raw/bureau.csv
Data/Raw/bureau_balance.csv
Data/Raw/previous_application.csv
Data/Raw/POS_CASH_balance.csv
Data/Raw/installments_payments.csv
Data/Raw/credit_card_balance.csv
```

`Data/` is ignored by git because it contains large raw and generated files.

## How To Run

Process raw data:

```powershell
.\.venv\Scripts\python.exe main.py --process
```

Train model and generate submission:

```powershell
.\.venv\Scripts\python.exe main.py --train
```

Run both stages:

```powershell
.\.venv\Scripts\python.exe main.py --all
```

Use a different config file:

```powershell
.\.venv\Scripts\python.exe main.py --all --config config.yaml
```

Important: if you change preprocessing options in `config.yaml`, run
`--process` before `--train`. Running only `--train` will reuse whatever
already exists in `Data/final/`.

## Run Profiles

Set `training.run_mode` in `config.yaml`.

| Profile | Purpose | CV Splits | Optuna Trials | Search Data | Full OOF |
| --- | --- | ---: | ---: | ---: | --- |
| `fast_dev` | Quick smoke/debug run | 2 | 5 | 15% | No |
| `balanced` | Practical speed/quality run | 3 | 10 | 35% | Yes |
| `balanced_deep` | Deeper search without full max cost | 3 | 30 | 50% | Yes |
| `max_quality` | Final scoring run | 5 | 50 | 100% | Yes |

The current config default is:

```yaml
training:
  run_mode: "max_quality"
  models:
    primary: "catboost"
```

`max_quality` is slow because it uses all training rows for search and 5-fold
OOF validation. Use `fast_dev` for quick checks.

## Outputs

Processing writes:

```text
Data/final/final_train.csv
Data/final/final_test.csv
Data/final/feature_manifest.yaml
```

Training writes one experiment folder directly under `Models/`:

```text
Models/<experiment_id>/
Models/latest_experiment.txt
```

Common experiment artifacts:

```text
config_snapshot.yaml
training_run_metadata.yaml
best_params.yaml
training_preprocessor.pkl
final_model.pkl
threshold.yaml
submission.csv
reports/evaluation_report.txt
reports/metrics.yaml
reports/oof_predictions.csv
reports/threshold_table.csv
reports/lift_deciles.csv
reports/feature_importance.csv
reports/calibration_diagnostics.yaml
plots/confusion_matrix.png
plots/roc_curve.png
plots/pr_curve.png
plots/calibration_curve.png
plots/lift_chart.png
plots/feature_importance_top.png
```

`Models/` is ignored by git because it contains generated models and reports.

## Configuration Guide

Most project behavior is controlled from `config.yaml`.

- `globals`: random seed, global epsilon for safe division, and job count.
- `data.csv`: CSV parser options, null tokens, and schema overrides.
- `data.raw`: raw Kaggle input file paths.
- `data.final`: processed train/test/manifest output paths.
- `pipeline`: preprocessing thresholds, fill values, categorical encoding,
  anomaly cleanup, feature-engineering sets, aggregation specs, recency windows,
  and last-N windows.
- `training.run_profiles`: runtime/quality profiles.
- `training.phases`: explicit `search`, `validate`, and `final_fit` switches.
- `training.experiment`: experiment folder naming.
- `training.threshold_tuning`: OOF/search-CV threshold tuning.
- `training.model_comparison`: optional single-model comparison reports.
- `training.calibration`: report-only calibration diagnostics.
- `training.acceleration`: GPU preference and CPU fallback.
- `training.preprocessing`: scaling, imbalance strategy, feature selection,
  and optional feature pruning.
- `training.models`: primary model, candidate model params, and Optuna search
  spaces.

## ML Logic Notes

- **OOF validation** means each train row is predicted by a model that did not
  train on that row.
- **Kaggle submission** uses raw probabilities, not thresholded labels.
- **Threshold tuning** is for reports/business classification metrics only.
- **Calibration diagnostics** measure probability quality; they do not alter
  submission probabilities because `apply_to_submission` is false.
- **GPU handling** tries the configured accelerator first and falls back to CPU
  for retryable GPU capability failures.
- **Artifact guards** save config/data hashes in metadata so stale artifact reuse
  can be detected.

## Null Handling

- CSV tokens such as `""`, `NA`, `NaN`, `nan`, and `NULL` are read as null.
- Important nullable application fields get `<column>_is_missing` indicators.
- Joined auxiliary-table nulls are filled with `pipeline.fill_values.aux_missing`
  because they usually mean "no related history row".
- Base numeric nulls are filled with train medians, and corresponding missing
  indicators are added when the train column had nulls.
- Categorical nulls become the configured `__NULL__` category for one-hot
  encoding.
- Generated ratio infinities/NaNs are converted to null and then filled with
  `pipeline.fill_values.generated_missing`.
- Final validation fails if train or test still contains nulls.

## File-By-File Map

### `.gitignore`

Ignores Python caches, virtual environments, generated data/model folders,
notebooks, CatBoost side-output, local environment files, and common build/test
artifacts.

### `.python-version`

Pins the intended Python version to `3.13`.

### `LICENSE`

MIT license for the project.

### `pyproject.toml`

Defines the package metadata and Python dependencies.

### `uv.lock`

Lockfile generated by `uv`; keeps dependency resolution reproducible.

### `config.yaml`

Single source of truth for paths, features, preprocessing behavior, model
selection, Optuna search spaces, run profiles, artifact names, diagnostics, and
training behavior.

### `main.py`

CLI entrypoint.

- `main`: parses `--process`, `--train`, `--all`, and `--config`; loads YAML
  config; calls the processing and/or training pipeline.

### `src/data_processing/run_pipeline.py`

Builds final train/test feature matrices.

- `_polars_dtype`: converts dtype names from config to Polars dtypes.
- `_csv_options`: builds Polars CSV reader options from config.
- `_safe_ratio_expr`: creates division expressions with configured epsilon.
- `_feature_enabled`: checks whether a feature set is enabled.
- `_available_lazy_columns`: lists columns in a lazy Polars frame.
- `_existing_columns`: filters requested columns to available columns.
- `_mean_agg_exprs`: builds mean aggregation expressions.
- `_sum_agg_exprs`: builds sum aggregation expressions.
- `_max_agg_exprs`: builds max aggregation expressions.
- `_last_n_filter`: keeps the most recent N records per customer/group.
- `_trend_expr`: creates recent-minus-global trend features.
- `load_data`: loads base train/test tables and lazy auxiliary tables.
- `get_proportions`: creates category proportion aggregations.
- `_sorted_unique_values`: gets sorted categorical values and optional null
  label.
- `_categorical_expr`: creates one-hot indicator expressions.
- `_binary_expr`: creates ordinal binary categorical encoding.
- `encode_categoricals`: encodes string categoricals using train-discovered
  categories.
- `apply_frequency_encoding`: replaces configured high-cardinality categoricals
  with train frequency counts.
- `preprocess_base`: fixes application anomalies, adds application-level ratios,
  document count, missing indicators, and categorical encodings.
  - `transform_base`: nested helper that applies those base-table transforms to
    either train or test.
- `agg_bureau`: aggregates bureau and bureau balance history, including
  active/closed splits and recency-window features.
- `agg_prev_app`: aggregates previous applications, interest-like ratios,
  status splits, and last-N features.
- `agg_pos_cash`: cleans configured POS invalid values and aggregates POS
  history, recency, last-N, and trend features.
- `agg_installments`: builds payment timing/ratio features and aggregates
  installment history.
- `agg_cc_balance`: builds credit-card utilization/receivable features and
  aggregates card history.
- `merge_all`: left-joins all auxiliary aggregations onto application rows and
  warns about high-null auxiliary columns.
- `impute_missing`: fills auxiliary nulls, median-fills base numeric nulls, and
  adds missing indicators.
- `add_global_features`: adds final global ratios, EXT_SOURCE summaries,
  enquiry totals, and cleans generated numeric NaN/inf values.
- `feature_cleanup`: drops correlated and low-variance features, aligns test to
  train columns, and records dropped columns.
- `validate`: checks target placement, null-free matrices, unique IDs, and
  train/test column alignment.
- `write_feature_manifest`: writes processing metadata and final column lists.
- `latest_submission_path`: resolves the latest known submission path for stale
  submission warnings.
- `run_pipeline`: orchestrates the full processing stage and writes final CSVs.

### `src/model_training/run_training.py`

Tunes, validates, fits, reports, and submits the configured model.

- `resolve_training_config`: applies the selected run profile into active
  training settings.
- `stable_yaml_hash`: hashes config content in stable YAML form.
- `file_hash`: hashes data files for artifact metadata.
- `slugify`: makes safe experiment folder names.
- `primary_model_name`: reads the configured primary model name.
- `get_primary_estimator_config`: returns the primary model config.
- `get_estimator_config_by_name`: finds a candidate model by name.
- `create_experiment_dir`: creates `Models/<experiment_id>/` with collision
  handling.
- `write_latest_experiment_pointer`: writes `Models/latest_experiment.txt`.
- `save_config_snapshot`: saves the effective config inside the experiment.
- `metadata_path`: resolves the run metadata path.
- `build_run_metadata`: builds config/data/run metadata.
- `write_run_metadata`: writes metadata and artifact list.
- `validate_reusable_artifacts`: prevents stale artifact reuse unless allowed.
- `clean_column_names`: sanitizes feature names for model libraries.
- `get_scaler`: creates the configured scaler.
- `parse_selector_threshold`: parses selector thresholds from config.
- `get_acceleration_config`: reads acceleration config for a model.
- `get_accelerator_order`: orders preferred/fallback accelerator attempts.
- `get_accelerator_params`: reads model-specific GPU/CPU params.
- `accelerator_failure_is_retryable`: checks whether an accelerator failure can
  fall back.
- `get_imbalance_config`: reads imbalance handling config.
- `get_imbalance_sampler`: creates an optional sampler, or returns none for
  class-weight mode.
- `split_speed_params`: separates training-speed fit options from model params.
- `merge_model_params`: merges base params, imbalance params, accelerator params,
  and verbosity.
- `fit_kwargs_for_model`: creates fold-local eval splits for early stopping.
- `capability_sample`: creates a tiny stratified sample for accelerator checks.
- `resolve_model_accelerator`: tests/caches whether GPU or CPU should be used.
- `fit_model`: fits one model with accelerator retry/fallback.
- `TrainingPreprocessor`: fold-safe scaler, selector, and optional pruning.
  - `__init__`: stores config and feature-selection flag.
  - `fit`: learns scaler, selector, and pruned columns from current train split.
  - `transform`: applies learned preprocessing and checks expected columns.
  - `fit_transform`: fits then transforms.
- `fit_lgbm_selector`: fits LightGBM feature selector.
- `fit_feature_pruning_columns`: chooses columns from selector importances when
  pruning is enabled.
- `calculate_metric`: computes ROC AUC, average precision, or F1.
- `model_artifact_path`: resolves experiment-relative artifact paths.
- `model_feature_importances`: extracts native model feature importance.
- `save_feature_importance`: saves feature-importance CSV and top-N plot.
- `threshold_grid`: builds the configured threshold grid.
- `build_threshold_table`: computes precision/recall/F1/accuracy per threshold.
- `choose_threshold`: selects threshold from OOF/search-CV predictions.
- `build_lift_table`: builds decile lift/capture diagnostics.
- `save_oof_predictions`: saves per-row OOF probabilities.
- `save_diagnostic_plots`: saves confusion matrix, ROC, PR, calibration, and
  lift plots.
- `calibrated_oof_predictions`: creates fold-safe isotonic-calibrated OOF
  probabilities for diagnostics.
- `save_calibration_diagnostics`: saves Brier-score calibration comparison.
- `save_evaluation_report`: writes metrics, threshold table, OOF predictions,
  lift table, plots, and text report.
- `suggest_params`: converts Optuna search-space config into trial suggestions.
- `merged_estimator_config`: returns a candidate config with param overrides.
- `get_cv`: creates stratified K-fold CV.
- `fit_predict_fold`: fits one fold and returns validation probabilities.
- `cross_validated_single_predictions`: builds complete OOF/search-CV
  probability predictions.
- `fit_final_single_model`: fits the final full-train model and saves artifacts.
- `configure_optuna_logging`: sets Optuna verbosity from config.
- `comparison_training_config`: creates a temporary config for optional model
  comparison.
- `model_comparison_sample`: samples data for model-comparison runs.
- `compare_one_model`: Optuna-tunes and scores one comparison candidate.
  - `objective`: nested Optuna objective used during model comparison.
- `run_model_comparison`: saves optional model comparison CSV/YAML.
- `run_training`: orchestrates training, metadata, comparison, phases, and latest
  pointer.
- `run_single_phases`: runs search, validation, and final-fit phases for the
  primary model.
- `run_single_search`: runs Optuna search and search-CV evaluation.
  - `objective`: nested Optuna objective used during primary-model search.
- `predict_test_and_submit`: transforms test data and writes Kaggle probabilities.

## Final Submission Notes

The submission file is:

```text
Models/<experiment_id>/submission.csv
```

It contains:

```text
SK_ID_CURR,TARGET
```

`TARGET` is a probability, not a hard 0/1 label. This is for Kaggle ROC
AUC scoring.
