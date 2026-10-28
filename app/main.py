from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, Sequence

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "predictive_maintenance_model.joblib"
)

EXPECTED_FEATURE_COLUMNS = {
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_H",
    "Type_L",
    "Type_M",
}


class ModelPredictor(Protocol):
    def predict(self, features: pd.DataFrame) -> Sequence[int]: ...


@dataclass(frozen=True)
class ModelBundle:
    model: ModelPredictor
    feature_columns: tuple[str, ...]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    air_temperature: float = Field(alias="Air temperature [K]", gt=0)
    process_temperature: float = Field(alias="Process temperature [K]", gt=0)
    rotational_speed: float = Field(alias="Rotational speed [rpm]", ge=0)
    torque: float = Field(alias="Torque [Nm]", ge=0)
    tool_wear: float = Field(alias="Tool wear [min]", ge=0)
    type_h: int = Field(alias="Type_H", ge=0, le=1)
    type_l: int = Field(alias="Type_L", ge=0, le=1)
    type_m: int = Field(alias="Type_M", ge=0, le=1)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    target: Literal[0, 1] = Field(alias="Target")
    label: Literal["No Failure", "Failure"]


@lru_cache(maxsize=1)
def load_model_bundle() -> ModelBundle:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Place predictive_maintenance_model.joblib in the models/ directory."
        )

    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict):
        raise ValueError("The model artifact must be a dictionary.")
    if "model" not in artifact or "feature_columns" not in artifact:
        raise ValueError(
            "The model artifact must contain 'model' and 'feature_columns'."
        )

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    if not callable(getattr(model, "predict", None)):
        raise ValueError("The model artifact's 'model' must provide predict().")
    if (
        not isinstance(feature_columns, (list, tuple))
        or not all(isinstance(column, str) for column in feature_columns)
        or len(feature_columns) != len(EXPECTED_FEATURE_COLUMNS)
        or set(feature_columns) != EXPECTED_FEATURE_COLUMNS
    ):
        raise ValueError(
            "The model artifact's feature_columns must contain exactly the "
            "eight supported predictive-maintenance features."
        )

    return ModelBundle(model=model, feature_columns=tuple(feature_columns))


def get_model_bundle() -> ModelBundle:
    try:
        return load_model_bundle()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


app = FastAPI(title="Predictive Maintenance API")


@app.get("/health")
def health_check(
    model_bundle: ModelBundle = Depends(get_model_bundle),
) -> dict[str, bool | str]:
    del model_bundle
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    data: PredictionRequest,
    model_bundle: ModelBundle = Depends(get_model_bundle),
) -> PredictionResponse:
    values = data.model_dump(by_alias=True)
    features = pd.DataFrame([values], columns=model_bundle.feature_columns)
    raw_prediction = model_bundle.model.predict(features)[0]
    if raw_prediction not in (0, 1):
        raise RuntimeError(
            f"Model returned unsupported Target value: {raw_prediction}"
        )
    prediction = int(raw_prediction)

    return PredictionResponse(
        Target=prediction,
        label="Failure" if prediction == 1 else "No Failure",
    )
