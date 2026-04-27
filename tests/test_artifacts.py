import tempfile
import unittest
from pathlib import Path

from src.common.artifacts import model_artifact_path
from src.model_training.artifacts import create_experiment_dir


class ModelArtifactPathTest(unittest.TestCase):
    def test_resolves_relative_artifact_inside_experiment_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {"training": {"artifact_paths": {"metrics": "reports/metrics.yaml"}}}

            path = model_artifact_path(tmp_dir, config, "metrics")

            self.assertEqual(path, Path(tmp_dir) / "reports" / "metrics.yaml")
            self.assertTrue(path.parent.exists())

    def test_resolves_template_artifact_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {"training": {"artifact_paths": {"model": "{model_name}.pkl"}}}

            path = model_artifact_path(tmp_dir, config, "model", model_name="catboost")

            self.assertEqual(path, Path(tmp_dir) / "catboost.pkl")


class ExperimentDirTest(unittest.TestCase):
    def test_named_experiment_uses_stable_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "training": {
                    "experiment": {
                        "name": "dvc_train",
                        "folder_template": "{timestamp}_{primary_model}",
                        "overwrite_existing": False,
                    },
                    "models": {"primary": "catboost"},
                }
            }

            path, experiment_id, _ = create_experiment_dir(Path(tmp_dir), config)

            self.assertEqual(path, Path(tmp_dir) / "dvc_train")
            self.assertEqual(experiment_id, "dvc_train")

    def test_named_experiment_can_overwrite_existing_folder(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            existing = Path(tmp_dir) / "dvc_train"
            existing.mkdir()
            stale_file = existing / "stale.txt"
            stale_file.write_text("old", encoding="utf-8")
            config = {
                "training": {
                    "experiment": {
                        "name": "dvc_train",
                        "folder_template": "{timestamp}_{primary_model}",
                        "overwrite_existing": True,
                    },
                    "models": {"primary": "catboost"},
                }
            }

            path, _, _ = create_experiment_dir(Path(tmp_dir), config)

            self.assertEqual(path, existing)
            self.assertFalse(stale_file.exists())


if __name__ == "__main__":
    unittest.main()
