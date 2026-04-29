from typing import Any

from pydantic import BaseModel, ConfigDict


ProcessedRow = dict[str, Any]


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    ready: bool
    model_source: str
    model_name: str
    model_alias: str
    model_uri: str
    tracking_uri: str | None = None
    error: str | None = None


class MetadataResponse(BaseModel):
    input_type: str
    id_col: str
    probability_col: str
    label_col: str
    include_binary_label: bool
    max_batch_size: int
    raw_input_warning: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    predictions: list[dict[str, Any]]
