import unittest
from types import SimpleNamespace

import pandas as pd

from src.api.model_service import ApiModelConfig, PredictionService, mlflow_model_uri


def model_config(max_batch_size=100, threshold=0.5):
    return ApiModelConfig(
        model_source="mlflow",
        model_name="home-credit-default-risk",
        model_alias="champion",
        id_col="SK_ID_CURR",
        probability_col="TARGET_PROBABILITY",
        label_col="TARGET_PREDICTION",
        include_binary_label=True,
        classification_threshold=threshold,
        max_batch_size=max_batch_size,
        tracking_uri="file:///tmp/mlruns",
    )


class FakeModel:
    def __init__(self, probabilities, signature_columns=None):
        self.probabilities = probabilities
        self.calls = 0
        self.last_frame = None
        if signature_columns is not None:
            if isinstance(signature_columns, dict):
                inputs = [SimpleNamespace(name=name, type=type_name) for name, type_name in signature_columns.items()]
            else:
                inputs = [SimpleNamespace(name=name) for name in signature_columns]
            self.metadata = SimpleNamespace(get_input_schema=lambda: SimpleNamespace(inputs=inputs))

    def predict(self, frame):
        self.calls += 1
        self.last_frame = frame.copy()
        return pd.DataFrame({"TARGET_PROBABILITY": self.probabilities[: len(frame)]})


class ApiModelServiceTest(unittest.TestCase):
    def test_mlflow_model_uri_uses_registry_alias(self):
        self.assertEqual(
            mlflow_model_uri("home-credit-default-risk", "champion"),
            "models:/home-credit-default-risk@champion",
        )

    def test_predict_preserves_ids_and_adds_threshold_label(self):
        model = FakeModel([0.25, 0.75])
        service = PredictionService(model_config(threshold=0.5), model=model)

        predictions = service.predict(
            [
                {"SK_ID_CURR": 1001, "FEATURE_A": 1.0},
                {"SK_ID_CURR": 1002, "FEATURE_A": 2.0},
            ]
        )

        self.assertEqual(model.calls, 1)
        self.assertEqual([item["SK_ID_CURR"] for item in predictions], [1001, 1002])
        self.assertEqual([item["TARGET_PREDICTION"] for item in predictions], [0, 1])
        self.assertEqual(predictions[0]["model_name"], "home-credit-default-risk")
        self.assertEqual(predictions[0]["model_alias"], "champion")

    def test_predict_uses_row_index_when_id_is_missing(self):
        service = PredictionService(model_config(), model=FakeModel([0.9]))

        predictions = service.predict([{"FEATURE_A": 1.0}])

        self.assertEqual(predictions[0]["SK_ID_CURR"], 0)

    def test_predict_aligns_missing_columns_to_model_signature(self):
        model = FakeModel([0.4], signature_columns=["SK_ID_CURR", "FEATURE_A", "FEATURE_B"])
        service = PredictionService(model_config(), model=model)

        service.predict([{"SK_ID_CURR": 10, "FEATURE_A": 1.5}])

        self.assertEqual(model.last_frame.columns.tolist(), ["SK_ID_CURR", "FEATURE_A", "FEATURE_B"])
        self.assertEqual(model.last_frame["FEATURE_B"].tolist(), [0])

    def test_predict_rejects_payload_with_no_recognized_features(self):
        model = FakeModel([0.4], signature_columns=["SK_ID_CURR", "FEATURE_A"])
        service = PredictionService(model_config(), model=model)

        with self.assertRaisesRegex(ValueError, "No recognized model feature"):
            service.predict([{"SK_ID_CURR": 10, "UNKNOWN_RAW_FIELD": 1.5}])

    def test_predict_casts_integer_values_to_float_schema_columns(self):
        model = FakeModel(
            [0.4],
            signature_columns={
                "SK_ID_CURR": "long",
                "AMT_INCOME_TOTAL": "double",
                "EXT_SOURCE_2": "double",
            },
        )
        service = PredictionService(model_config(), model=model)

        service.predict([{"SK_ID_CURR": 10, "AMT_INCOME_TOTAL": 10000, "EXT_SOURCE_2": 1}])

        self.assertEqual(str(model.last_frame["AMT_INCOME_TOTAL"].dtype), "float64")
        self.assertEqual(str(model.last_frame["EXT_SOURCE_2"].dtype), "float64")

    def test_predict_rejects_oversized_batch(self):
        service = PredictionService(model_config(max_batch_size=1), model=FakeModel([0.1, 0.2]))

        with self.assertRaisesRegex(ValueError, "max_batch_size"):
            service.predict([{"FEATURE_A": 1.0}, {"FEATURE_A": 2.0}])


if __name__ == "__main__":
    unittest.main()
