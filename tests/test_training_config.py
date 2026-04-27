import unittest

from src.common.config_io import load_hydra_config


class TrainingConfigTest(unittest.TestCase):
    def test_training_defaults_match_config_values(self):
        config = load_hydra_config()
        training = config["training"]

        self.assertEqual(training["cv_splits"], 3)
        self.assertEqual(training["optuna_n_trials"], 5)
        self.assertEqual(training["optuna_subsample_rate"], 0.20)
        self.assertTrue(training["run_full_oof_validation"])
        self.assertFalse(training["preprocessing"]["feature_selection"]["enabled_during_search"])

    def test_training_params_are_directly_overridable(self):
        config = load_hydra_config(
            [
                "training.cv_splits=2",
                "training.optuna_n_trials=5",
                "training.optuna_subsample_rate=0.15",
                "training.run_full_oof_validation=false",
            ]
        )
        training = config["training"]

        self.assertEqual(training["cv_splits"], 2)
        self.assertEqual(training["optuna_n_trials"], 5)
        self.assertEqual(training["optuna_subsample_rate"], 0.15)
        self.assertFalse(training["run_full_oof_validation"])


if __name__ == "__main__":
    unittest.main()
