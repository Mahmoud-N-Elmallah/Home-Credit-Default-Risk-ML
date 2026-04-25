import tempfile
import unittest
from pathlib import Path

from src.common.artifacts import model_artifact_path


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


if __name__ == "__main__":
    unittest.main()
