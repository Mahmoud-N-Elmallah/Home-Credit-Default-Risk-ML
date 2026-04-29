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


@dataclass(frozen=True)
class InputDiagnostics:
    recognized_columns: list[str]
    ignored_columns: list[str]
    filled_missing_count: int
    filled_missing_preview: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "recognized_columns": self.recognized_columns,
            "ignored_columns": self.ignored_columns,
            "filled_missing_count": self.filled_missing_count,
            "filled_missing_preview": self.filled_missing_preview,
        }


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

    def predict_with_diagnostics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        predictions, diagnostics = self._predict(rows)
        return {"predictions": predictions, "input_diagnostics": diagnostics.as_dict()}

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        predictions, _ = self._predict(rows)
        return predictions

    def _predict(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], InputDiagnostics]:
        if self.model is None:
            raise ModelNotLoadedError("Model is not loaded.")
        if not rows:
            raise ValueError("Prediction payload cannot be empty.")
        if len(rows) > self.config.max_batch_size:
            raise ValueError(f"Prediction batch exceeds api.max_batch_size={self.config.max_batch_size}.")

        frame = pd.DataFrame(rows)
        ids = frame[self.config.id_col].tolist() if self.config.id_col in frame.columns else list(range(len(frame)))
        frame, diagnostics = align_to_model_signature(self.model, frame, self.config.id_col)
        try:
            predictions = self.model.predict(frame)
        except Exception as exc:
            raise ValueError(f"Model prediction failed: {exc}") from exc
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
        return output, diagnostics


def align_to_model_signature(model, frame: pd.DataFrame, id_col: str) -> tuple[pd.DataFrame, InputDiagnostics]:
    expected_schema = model_signature_schema(model)
    if not expected_schema:
        return frame, InputDiagnostics(list(frame.columns), [], 0, [])

    expected_columns = list(expected_schema.keys())
    provided_columns = list(frame.columns)
    expected_set = set(expected_columns)
    recognized_columns = [column for column in provided_columns if column in expected_set]
    ignored_columns = [column for column in provided_columns if column not in expected_set]
    recognized_features = [column for column in recognized_columns if column != id_col]
    if not recognized_features:
        raise ValueError(
            "No recognized model feature columns were provided. Send processed feature columns from /metadata, "
            "for example EXT_SOURCE_2, EXT_SOURCE_3, EXT_SOURCES_MEAN, or CREDIT_TERM."
        )

    missing_columns = [column for column in expected_columns if column not in frame.columns]
    aligned = frame.reindex(columns=expected_columns, fill_value=0)
    for column, type_name in expected_schema.items():
        aligned[column] = coerce_column_to_schema_type(aligned[column], type_name, column)

    diagnostics = InputDiagnostics(
        recognized_columns=recognized_columns,
        ignored_columns=ignored_columns,
        filled_missing_count=len(missing_columns),
        filled_missing_preview=missing_columns[:10],
    )
    return aligned, diagnostics


def model_signature_columns(model) -> list[str]:
    return list(model_signature_schema(model).keys())


def model_signature_schema(model) -> dict[str, str]:
    metadata = getattr(model, "metadata", None)
    if metadata is None or not hasattr(metadata, "get_input_schema"):
        return {}
    schema = metadata.get_input_schema()
    if schema is None:
        return {}
    columns = {}
    for input_spec in getattr(schema, "inputs", []):
        name = getattr(input_spec, "name", None)
        if name:
            columns[name] = str(getattr(input_spec, "type", "")).lower()
    return columns


def coerce_column_to_schema_type(series: pd.Series, type_name: str, column: str) -> pd.Series:
    try:
        if "double" in type_name or "float" in type_name:
            return pd.to_numeric(series, errors="raise").astype("float64")
        if "long" in type_name or "integer" in type_name:
            return pd.to_numeric(series, errors="raise").astype("int64")
    except Exception as exc:
        raise ValueError(f"Column {column} cannot be converted to MLflow schema type {type_name}.") from exc
    return series


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
