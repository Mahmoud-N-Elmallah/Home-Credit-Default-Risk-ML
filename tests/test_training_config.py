import unittest

from src.model_training.config import resolve_training_config


class ResolveTrainingConfigTest(unittest.TestCase):
    def test_applies_selected_run_profile(self):
        config = {
            "training": {
                "run_mode": "fast_dev",
                "run_profiles": {
                    "fast_dev": {
                        "cv_splits": 2,
                        "optuna_n_trials": 5,
                        "optuna_subsample_rate": 0.15,
                        "run_full_oof_validation": False,
                        "feature_selection_enabled_during_search": False,
                    }
                },
                "preprocessing": {
                    "feature_selection": {
                        "enabled_during_search": True,
                    }
                },
            }
        }

        resolved = resolve_training_config(config)

        self.assertEqual(resolved["training"]["cv_splits"], 2)
        self.assertEqual(resolved["training"]["optuna_n_trials"], 5)
        self.assertEqual(resolved["training"]["optuna_subsample_rate"], 0.15)
        self.assertFalse(resolved["training"]["run_full_oof_validation"])
        self.assertFalse(
            resolved["training"]["preprocessing"]["feature_selection"]["enabled_during_search"]
        )
        self.assertTrue(
            config["training"]["preprocessing"]["feature_selection"]["enabled_during_search"]
        )

    def test_unknown_profile_errors(self):
        config = {
            "training": {
                "run_mode": "missing",
                "run_profiles": {},
            }
        }

        with self.assertRaises(ValueError):
            resolve_training_config(config)


if __name__ == "__main__":
    unittest.main()
