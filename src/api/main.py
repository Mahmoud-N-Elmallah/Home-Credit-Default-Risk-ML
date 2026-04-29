from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.api.model_service import ModelNotLoadedError, PredictionService, api_model_config
from src.api.schemas import HealthResponse, MetadataResponse, PredictionResponse, ReadinessResponse
from src.common.config_io import load_hydra_config


def create_app(config: dict[str, Any] | None = None, service: PredictionService | None = None) -> FastAPI:
    app_config = load_hydra_config() if config is None else config
    prediction_service = service or PredictionService(api_model_config(app_config))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.prediction_service = prediction_service.load() if service is None else prediction_service
        yield

    app = FastAPI(title="Home Credit Default Risk API", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health():
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadinessResponse)
    def ready(request: Request):
        active_service = request.app.state.prediction_service
        payload = readiness_payload(active_service)
        if not active_service.ready:
            return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)
        return payload

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata(request: Request):
        active_service = request.app.state.prediction_service
        cfg = active_service.config
        return {
            "input_type": "processed_features",
            "id_col": cfg.id_col,
            "probability_col": cfg.probability_col,
            "label_col": cfg.label_col,
            "include_binary_label": cfg.include_binary_label,
            "max_batch_size": cfg.max_batch_size,
            "raw_input_warning": "Raw application rows are not accepted; send processed feature rows matching Data/final schema.",
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        request: Request,
        payload: Any = Body(
            ...,
            examples=[
                {
                    "summary": "Single processed row",
                    "value": {"SK_ID_CURR": 100001, "EXT_SOURCE_2": 0.5, "EXT_SOURCE_3": 0.4},
                },
                {
                    "summary": "Batch processed rows",
                    "value": [
                        {"SK_ID_CURR": 100001, "EXT_SOURCE_2": 0.5, "EXT_SOURCE_3": 0.4},
                        {"SK_ID_CURR": 100002, "EXT_SOURCE_2": 0.2, "EXT_SOURCE_3": 0.8},
                    ],
                },
            ],
        ),
    ):
        rows = normalize_payload(payload)
        active_service = request.app.state.prediction_service
        try:
            return {"predictions": active_service.predict(rows)}
        except ModelNotLoadedError as error:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    return app


def readiness_payload(service: PredictionService) -> dict[str, Any]:
    return {
        "ready": service.ready,
        "model_source": service.config.model_source,
        "model_name": service.config.model_name,
        "model_alias": service.config.model_alias,
        "model_uri": service.model_uri,
        "tracking_uri": service.config.tracking_uri,
        "error": service.load_error,
    }


def normalize_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and payload and all(isinstance(item, dict) for item in payload):
        return payload
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Prediction payload must be a JSON object or a non-empty array of JSON objects.",
    )


app = create_app()
