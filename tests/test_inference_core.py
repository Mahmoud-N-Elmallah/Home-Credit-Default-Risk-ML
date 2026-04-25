import unittest
from types import SimpleNamespace

import pandas as pd

from src.inference.core import prepare_features


def config(missing_feature_strategy="fill", max_missing_features_to_fill=20):
    return {
        "training": {"id_col": "SK_ID_CURR", "target_col": "TARGET"},
        "inference": {
            "allow_target_column": True,
            "missing_feature_strategy": missing_feature_strategy,
            "missing_feature_fill_value": 0,
            "max_missing_features_to_fill": max_missing_features_to_fill,
        },
    }


class PrepareFeaturesTest(unittest.TestCase):
    def test_aligns_missing_and_extra_columns_to_preprocessor_schema(self):
        preprocessor = SimpleNamespace(
            scaler=SimpleNamespace(feature_names_in_=["A", "B"]),
            selector=None,
        )
        frame = pd.DataFrame({"SK_ID_CURR": [10], "A": [1.5], "EXTRA": [99], "TARGET": [0]})

        ids, features, alignment = prepare_features(frame, preprocessor, config())

        self.assertEqual(ids.tolist(), [10])
        self.assertEqual(features.columns.tolist(), ["A", "B"])
        self.assertEqual(features["B"].tolist(), [0])
        self.assertEqual(alignment["missing_columns"], ["B"])
        self.assertEqual(alignment["extra_columns"], ["EXTRA"])

    def test_errors_when_missing_feature_strategy_is_error(self):
        preprocessor = SimpleNamespace(
            scaler=SimpleNamespace(feature_names_in_=["A", "B"]),
            selector=None,
        )
        frame = pd.DataFrame({"SK_ID_CURR": [10], "A": [1.5]})

        with self.assertRaises(ValueError):
            prepare_features(frame, preprocessor, config(missing_feature_strategy="error"))


if __name__ == "__main__":
    unittest.main()
