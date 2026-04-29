import unittest

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.model_service import PredictionService
from tests.test_api_model_service import FakeModel, model_config


def api_test_client(service):
    return TestClient(create_app(config={}, service=service))


class ApiMainTest(unittest.TestCase):
    def test_health_works_without_loaded_model(self):
        service = PredictionService(model_config())

        with api_test_client(service) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_ready_returns_503_when_model_failed_to_load(self):
        service = PredictionService(model_config())
        service.load_error = "load failed"

        with api_test_client(service) as client:
            response = client.get("/ready")

        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["ready"])
        self.assertEqual(response.json()["error"], "load failed")

    def test_metadata_reports_processed_feature_contract(self):
        service = PredictionService(model_config(), model=FakeModel([0.1]))

        with api_test_client(service) as client:
            response = client.get("/metadata")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["input_type"], "processed_features")
        self.assertEqual(payload["id_col"], "SK_ID_CURR")
        self.assertIn("Raw application rows are not accepted", payload["raw_input_warning"])

    def test_predict_accepts_single_row(self):
        service = PredictionService(model_config(), model=FakeModel([0.7]))

        with api_test_client(service) as client:
            response = client.post("/predict", json={"SK_ID_CURR": 42, "FEATURE_A": 1.0})

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["predictions"][0]["SK_ID_CURR"], 42)
        self.assertEqual(payload["predictions"][0]["TARGET_PROBABILITY"], 0.7)

    def test_predict_declares_request_body_for_swagger(self):
        service = PredictionService(model_config(), model=FakeModel([0.7]))

        with api_test_client(service) as client:
            schema = client.get("/openapi.json").json()
            response = client.post("/predict")

        self.assertIn("requestBody", schema["paths"]["/predict"]["post"])
        self.assertEqual(response.status_code, 422)

    def test_predict_accepts_batch_rows(self):
        service = PredictionService(model_config(), model=FakeModel([0.2, 0.8]))

        with api_test_client(service) as client:
            response = client.post(
                "/predict",
                json=[
                    {"SK_ID_CURR": 10, "FEATURE_A": 1.0},
                    {"SK_ID_CURR": 11, "FEATURE_A": 2.0},
                ],
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["predictions"]), 2)
        self.assertEqual(service.model.calls, 1)

    def test_predict_rejects_empty_scalar_and_oversized_payloads(self):
        service = PredictionService(model_config(max_batch_size=1), model=FakeModel([0.1, 0.2]))

        with api_test_client(service) as client:
            empty_response = client.post("/predict", json=[])
            scalar_response = client.post("/predict", json=123)
            oversized_response = client.post(
                "/predict",
                json=[
                    {"FEATURE_A": 1.0},
                    {"FEATURE_A": 2.0},
                ],
            )

        self.assertEqual(empty_response.status_code, 400)
        self.assertEqual(scalar_response.status_code, 400)
        self.assertEqual(oversized_response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
