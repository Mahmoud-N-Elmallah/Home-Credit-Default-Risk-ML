import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlflow.pyfunc

from src.model_training.tracking import CreditRiskPyFuncModel
from src.model_training.tracking import (
    MlflowTracker,
    configure_tracking_backend,
    dagshub_repo_from_uri,
    dvc_remote_url,
    end_mlflow_run,
    numeric_items,
    tracking_run,
)


def tracking_config(enabled=False):
    return {
        "tracking": {
            "mlflow": {
                "enabled": enabled,
                "log_artifacts": True,
                "log_model_artifacts": True,
                "registry": {
                    "enabled": False,
                    "registered_model_name": "home-credit-default-risk",
                    "alias": "champion",
                    "min_roc_auc": 0.0,
                    "required": False,
                },
            }
        },
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
        "data": {"final": {"train": "Data/final/final_train.csv"}},
        "inference": {"probability_col": "TARGET_PROBABILITY"},
    }


class FakeMlflow:
    def __init__(self):
        self.artifacts = []
        self.metrics = {}
        self.params = {}
        self.tags = {}
        self.model_version_tags = {}
        self.aliases = {}
        self.fail_alias = False
        self.tracking = SimpleNamespace(MlflowClient=lambda: self)

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

    def set_tag(self, key, value):
        self.tags[key] = value

    def set_model_version_tag(self, model_name, version, key, value):
        self.model_version_tags[(model_name, version, key)] = value

    def set_registered_model_alias(self, model_name, alias, version):
        if self.fail_alias:
            raise RuntimeError("alias API unavailable")
        self.aliases[(model_name, alias)] = version


class RegistryTracker(MlflowTracker):
    def __init__(self, mlflow_module, config, models_dir):
        super().__init__(mlflow_module, config, models_dir)
        self.registry_calls = []

    def _log_pyfunc_model(self, registry):
        self.registry_calls.append(registry)
        return SimpleNamespace(registered_model_version="7")


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

    def test_tracking_backend_loads_project_dotenv(self):
        config = {"tracking_uri": "file:///tmp/mlruns"}

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("src.model_training.tracking.load_project_dotenv") as load_env,
            patch("mlflow.set_tracking_uri") as set_tracking_uri,
        ):
            configure_tracking_backend(config)

        load_env.assert_called_once()
        set_tracking_uri.assert_called_once_with("file:///tmp/mlruns")

    def test_tracking_backend_prefers_env_tracking_uri(self):
        config = {"tracking_uri": "file:///tmp/config-mlruns"}

        with (
            patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "file:///tmp/env-mlruns"}, clear=True),
            patch("src.model_training.tracking.load_project_dotenv"),
            patch("mlflow.set_tracking_uri") as set_tracking_uri,
        ):
            configure_tracking_backend(config)

        set_tracking_uri.assert_called_once_with("file:///tmp/env-mlruns")

    def test_dvc_remote_url_prefers_env_url(self):
        with (
            patch.dict("os.environ", {"DVC_REMOTE_URL": "https://example.test/project.dvc"}, clear=True),
            patch("src.model_training.tracking.load_project_dotenv"),
        ):
            self.assertEqual(dvc_remote_url(), "https://example.test/project.dvc")

    def test_registry_wrapper_is_mlflow_python_model(self):
        self.assertTrue(issubclass(CreditRiskPyFuncModel, mlflow.pyfunc.PythonModel))

    def test_end_mlflow_run_ignores_terminal_encoding_error(self):
        fake = SimpleNamespace(end_run=lambda status: (_ for _ in ()).throw(UnicodeEncodeError("cp1252", "", 0, 1, "x")))

        end_mlflow_run(fake, status="FINISHED")

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

            self.assertEqual(fake.metrics["ranking.roc_auc"], 0.81)
            self.assertIn(("metrics.yaml", "reports"), fake.artifacts)

    def test_start_logging_records_params_and_tags_before_metrics_exist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir)
            metadata = {
                "experiment_id": "x",
                "primary_model": "catboost",
                "config_hash": "abc",
                "data_hashes": {"train": "hash"},
            }
            fake = FakeMlflow()

            MlflowTracker(fake, tracking_config(enabled=True), models_dir).log_start(metadata)

            self.assertEqual(fake.params["primary_model"], "catboost")
            self.assertEqual(fake.tags["experiment_id"], "x")
            self.assertEqual(fake.metrics, {})

    def test_registry_skips_when_metric_gate_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir)
            (models_dir / "reports").mkdir()
            (models_dir / "reports" / "metrics.yaml").write_text("ranking:\n  roc_auc: 0.5\n", encoding="utf-8")
            config = tracking_config(enabled=True)
            config["tracking"]["mlflow"]["registry"]["enabled"] = True
            config["tracking"]["mlflow"]["registry"]["min_roc_auc"] = 0.8
            fake = FakeMlflow()

            RegistryTracker(fake, config, models_dir).log_final(
                {"experiment_id": "x", "primary_model": "catboost", "config_hash": "abc"}
            )

            self.assertEqual(fake.tags["registry_status"], "skipped_metric_gate")

    def test_registry_registers_model_and_sets_alias(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir)
            (models_dir / "reports").mkdir()
            (models_dir / "reports" / "metrics.yaml").write_text("ranking:\n  roc_auc: 0.9\n", encoding="utf-8")
            config = tracking_config(enabled=True)
            config["tracking"]["mlflow"]["registry"]["enabled"] = True
            fake = FakeMlflow()
            metadata = {
                "experiment_id": "x",
                "primary_model": "catboost",
                "config_hash": "abc",
                "data_hashes": {"train": "hash"},
            }

            tracker = RegistryTracker(fake, config, models_dir)
            tracker.log_final(metadata)

            self.assertEqual(fake.tags["registry_status"], "registered")
            self.assertEqual(fake.aliases[("home-credit-default-risk", "champion")], "7")
            self.assertEqual(
                fake.model_version_tags[("home-credit-default-risk", "7", "experiment_id")],
                "x",
            )
            self.assertEqual(len(tracker.registry_calls), 1)

    def test_registry_stays_registered_when_alias_api_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = Path(tmp_dir)
            (models_dir / "reports").mkdir()
            (models_dir / "reports" / "metrics.yaml").write_text("ranking:\n  roc_auc: 0.9\n", encoding="utf-8")
            config = tracking_config(enabled=True)
            config["tracking"]["mlflow"]["registry"]["enabled"] = True
            fake = FakeMlflow()
            fake.fail_alias = True

            RegistryTracker(fake, config, models_dir).log_final(
                {"experiment_id": "x", "primary_model": "catboost", "config_hash": "abc"}
            )

            self.assertEqual(fake.tags["registry_status"], "registered")


if __name__ == "__main__":
    unittest.main()
