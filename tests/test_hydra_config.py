import unittest

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


if __name__ == "__main__":
    unittest.main()
