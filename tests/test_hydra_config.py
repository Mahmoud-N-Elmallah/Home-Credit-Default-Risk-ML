import unittest
from unittest.mock import patch

from src.common.config_io import load_hydra_config


class HydraConfigTest(unittest.TestCase):
    def test_conf_config_is_source_of_truth(self):
        config = load_hydra_config(["run.step=process"])

        self.assertEqual(config["run"]["step"], "process")
        self.assertIn("training", config)
        self.assertNotIn("baseline", config)
        self.assertNotIn("schema_overrides", config["data"]["csv"])

    def test_fixed_feature_constants_are_not_configured(self):
        config = load_hydra_config()

        feature_config = config["pipeline"]["feature_engineering"]
        self.assertIn("enabled_sets", feature_config)
        self.assertNotIn("base_ratio_features", feature_config)
        self.assertNotIn("recency_months", feature_config)
        self.assertNotIn("aggregations", config["pipeline"])

    def test_fixed_runtime_details_are_not_configured(self):
        config = load_hydra_config()
        training = config["training"]
        shap_config = config["analysis"]["shap"]

        self.assertNotIn("acceleration", training)
        self.assertNotIn("evaluation", training)
        self.assertNotIn("reports", training)
        self.assertNotIn("verbosity", training)
        self.assertEqual(list(training["artifact_paths"].keys()), ["models_dir"])
        self.assertEqual(set(shap_config.keys()), {"sample_size", "top_n"})

    def test_mlflow_tracking_uri_can_come_from_environment(self):
        with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "file:///tmp/test-mlruns"}):
            config = load_hydra_config()

        self.assertEqual(config["tracking"]["mlflow"]["tracking_uri"], "file:///tmp/test-mlruns")

    def test_registry_only_accepts_models_above_roc_auc_gate(self):
        config = load_hydra_config()

        self.assertEqual(config["tracking"]["mlflow"]["registry"]["min_roc_auc"], 0.785)


if __name__ == "__main__":
    unittest.main()
