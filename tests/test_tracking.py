import tempfile
import unittest
from pathlib import Path

from src.model_training.tracking import MlflowTracker, dagshub_repo_from_uri, numeric_items, tracking_run


def tracking_config(enabled=False):
    return {
        "tracking": {"mlflow": {"enabled": enabled, "log_artifacts": True, "log_model_artifacts": True}},
        "training": {
            "cv_splits": 3,
            "optuna_n_trials": 10,
            "optuna_subsample_rate": 0.35,
            "run_full_oof_validation": True,
            "optimization_metric": "roc_auc",
            "cv_shuffle": True,
            "phases": {"search": True, "validate": True, "final_fit": True},
            "threshold_tuning": {"objective": "f1"},
            "accelerator": "gpu",
            "preprocessing": {
                "scaler": "robust",
                "imbalance": {"strategy": "class_weight"},
                "feature_selection": {"enabled_during_search": False, "max_features": 150},
            },
            "artifact_reuse": {},
            "models": {
                "primary": "catboost",
                "candidates": [{"name": "catboost", "params": {"depth": 6}}],
            },
        },
    }


class FakeMlflow:
    def __init__(self):
        self.artifacts = []
        self.metrics = {}
        self.params = {}
        self.tags = {}

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((Path(path).name, artifact_path))

    def log_artifacts(self, path, artifact_path=None):
        self.artifacts.append((Path(path).name, artifact_path))

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def log_params(self, params):
        self.params.update(params)

    def set_tags(self, tags):
        self.tags.update(tags)


class TrackingTest(unittest.TestCase):
    def test_disabled_tracking_context_is_noop(self):
        with tracking_run(tracking_config(enabled=False), ".", {"experiment_id": "x"}) as tracker:
            tracker.log_final({})

    def test_numeric_items_flattens_only_numbers(self):
        metrics = {"ranking": {"roc_auc": 0.8}, "name": "model", "flag": True}

        self.assertEqual(dict(numeric_items(metrics)), {"ranking.roc_auc": 0.8})

    def test_dagshub_repo_from_uri_parses_tracking_uri(self):
        uri = "https://dagshub.com/mahmoudelmalah85/Home-Credit-Default-Risk-ML.mlflow"

        self.assertEqual(
            dagshub_repo_from_uri(uri),
            ("mahmoudelmalah85", "Home-Credit-Default-Risk-ML"),
        )

    def test_artifact_logging_skips_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir)
            (models_dir / "config_snapshot.yaml").write_text("x: 1\n", encoding="utf-8")
            (models_dir / "training_run_metadata.yaml").write_text("experiment_id: x\n", encoding="utf-8")
            fake = FakeMlflow()

            MlflowTracker(fake, tracking_config(enabled=True), models_dir)._log_artifacts()

            logged_names = [name for name, _ in fake.artifacts]
            self.assertEqual(logged_names, ["config_snapshot.yaml", "training_run_metadata.yaml"])

    def test_final_logging_records_params_tags_metrics_and_curated_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir)
            (models_dir / "reports").mkdir()
            (models_dir / "config_snapshot.yaml").write_text("x: 1\n", encoding="utf-8")
            (models_dir / "reports" / "metrics.yaml").write_text("ranking:\n  roc_auc: 0.81\n", encoding="utf-8")
            (models_dir / "training_run_metadata.yaml").write_text("experiment_id: x\n", encoding="utf-8")
            metadata = {
                "experiment_id": "x",
                "primary_model": "catboost",
                "config_hash": "abc",
                "data_hashes": {"train": "hash"},
            }
            fake = FakeMlflow()

            MlflowTracker(fake, tracking_config(enabled=True), models_dir).log_final(metadata)

            self.assertEqual(fake.params["primary_model"], "catboost")
            self.assertEqual(fake.tags["experiment_id"], "x")
            self.assertEqual(fake.metrics["ranking.roc_auc"], 0.81)
            self.assertIn(("metrics.yaml", "reports"), fake.artifacts)


if __name__ == "__main__":
    unittest.main()
