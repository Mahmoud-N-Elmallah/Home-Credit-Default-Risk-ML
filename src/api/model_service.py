import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.common.env import load_project_dotenv
from src.model_training.tracking import configure_tracking_backend


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiModelConfig:
    model_source: str
    model_name: str
    model_alias: str
    id_col: str
    probability_col: str
    label_col: str
    include_binary_label: bool
    classification_threshold: float
    max_batch_size: int
    tracking_uri: str | None = None


class ModelNotLoadedError(RuntimeError):
    pass


def api_model_config(config: dict[str, Any]) -> ApiModelConfig:
    api_config = config["api"]
    inference_config = config["inference"]
    tracking_config = config.get("tracking", {}).get("mlflow", {})
    return ApiModelConfig(
        model_source=str(api_config["model_source"]),
        model_name=str(api_config["model_name"]),
        model_alias=str(api_config["model_alias"]),
        id_col=str(config["training"]["id_col"]),
        probability_col=str(inference_config["probability_col"]),
        label_col=str(inference_config["label_col"]),
        include_binary_label=bool(api_config["include_binary_label"]),
        classification_threshold=float(api_config["classification_threshold"]),
        max_batch_size=int(api_config["max_batch_size"]),
        tracking_uri=tracking_config.get("tracking_uri"),
    )


def mlflow_model_uri(model_name: str, model_alias: str) -> str:
    return f"models:/{model_name}@{model_alias}"


def load_mlflow_model(model_config: ApiModelConfig):
    if model_config.model_source != "mlflow":
        raise ValueError(f"Unsupported api.model_source: {model_config.model_source}")

    load_project_dotenv()
    import mlflow

    if model_config.tracking_uri:
        configure_tracking_backend({"tracking_uri": model_config.tracking_uri})
    model_uri = mlflow_model_uri(model_config.model_name, model_config.model_alias)
    logger.info("Loading MLflow model from %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)


class PredictionService:
    def __init__(self, model_config: ApiModelConfig, model=None):
        self.config = model_config
        self.model = model
        self.load_error: str | None = None

    @property
    def model_uri(self) -> str:
        return mlflow_model_uri(self.config.model_name, self.config.model_alias)

    @property
    def ready(self) -> bool:
        return self.model is not None and self.load_error is None

    def load(self):
        try:
            self.model = load_mlflow_model(self.config)
            self.load_error = None
        except Exception as error:
            self.model = None
            self.load_error = str(error)
            logger.exception("Failed to load API model.")
        return self

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.model is None:
            raise ModelNotLoadedError("Model is not loaded.")
        if not rows:
            raise ValueError("Prediction payload cannot be empty.")
        if len(rows) > self.config.max_batch_size:
            raise ValueError(f"Prediction batch exceeds api.max_batch_size={self.config.max_batch_size}.")

        frame = pd.DataFrame(rows)
        ids = frame[self.config.id_col].tolist() if self.config.id_col in frame.columns else list(range(len(frame)))
        frame = align_to_model_signature(self.model, frame)
        predictions = self.model.predict(frame)
        probabilities = prediction_probabilities(predictions, self.config.probability_col)

        output = []
        for row_id, probability in zip(ids, probabilities, strict=True):
            item = {
                self.config.id_col: row_id,
                self.config.probability_col: float(probability),
                "model_name": self.config.model_name,
                "model_alias": self.config.model_alias,
            }
            if self.config.include_binary_label:
                item[self.config.label_col] = int(float(probability) >= self.config.classification_threshold)
            output.append(item)
        return output


def align_to_model_signature(model, frame: pd.DataFrame) -> pd.DataFrame:
    expected_columns = model_signature_columns(model)
    if not expected_columns:
        return frame
    return frame.reindex(columns=expected_columns, fill_value=0)


def model_signature_columns(model) -> list[str]:
    metadata = getattr(model, "metadata", None)
    if metadata is None or not hasattr(metadata, "get_input_schema"):
        return []
    schema = metadata.get_input_schema()
    if schema is None:
        return []
    columns = []
    for input_spec in getattr(schema, "inputs", []):
        name = getattr(input_spec, "name", None)
        if name:
            columns.append(name)
    return columns


def prediction_probabilities(predictions, probability_col: str) -> list[float]:
    if isinstance(predictions, pd.DataFrame):
        if probability_col not in predictions.columns:
            raise ValueError(f"Model output is missing probability column: {probability_col}")
        return predictions[probability_col].astype(float).tolist()
    if isinstance(predictions, pd.Series):
        return predictions.astype(float).tolist()
    if isinstance(predictions, list):
        return [float(value) for value in predictions]

    try:
        values = predictions.tolist()
    except AttributeError as exc:
        raise ValueError("Unsupported model prediction output type.") from exc
    if values and isinstance(values[0], list):
        values = [row[0] for row in values]
    return [float(value) for value in values]
