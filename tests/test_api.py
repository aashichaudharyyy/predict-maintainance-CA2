from collections.abc import Sequence

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import (
    EXPECTED_FEATURE_COLUMNS,
    MODEL_PATH,
    ModelBundle,
    app,
    load_model_bundle,
    get_model_bundle,
)


class StubModel:
    def __init__(self, prediction: int = 1) -> None:
        self.prediction = prediction
        self.received_columns: list[str] | None = None

    def predict(self, features: pd.DataFrame) -> Sequence[int]:
        self.received_columns = list(features.columns)
        return [self.prediction]


@pytest.fixture
def client():
    model = StubModel()
    bundle = ModelBundle(
        model=model,
        feature_columns=(
            "Type_M",
            "Air temperature [K]",
            "Tool wear [min]",
            "Type_H",
            "Torque [Nm]",
            "Rotational speed [rpm]",
            "Type_L",
            "Process temperature [K]",
        ),
    )
    app.dependency_overrides[get_model_bundle] = lambda: bundle
    with TestClient(app) as test_client:
        yield test_client, model
    app.dependency_overrides.clear()


@pytest.fixture
def valid_payload() -> dict[str, float | int]:
    return {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 0,
        "Type_H": 0,
        "Type_L": 1,
        "Type_M": 0,
    }


def test_health_endpoint(client):
    test_client, _ = client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_predicts_failure_and_uses_artifact_feature_order(client, valid_payload):
    test_client, model = client

    response = test_client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json() == {"Target": 1, "label": "Failure"}
    assert model.received_columns == [
        "Type_M",
        "Air temperature [K]",
        "Tool wear [min]",
        "Type_H",
        "Torque [Nm]",
        "Rotational speed [rpm]",
        "Type_L",
        "Process temperature [K]",
    ]


def test_predicts_no_failure(client, valid_payload):
    test_client, model = client
    model.prediction = 0

    response = test_client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json() == {"Target": 0, "label": "No Failure"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Air temperature [K]", 0),
        ("Type_H", 2),
    ],
)
def test_rejects_invalid_feature_values(client, valid_payload, field, value):
    test_client, _ = client
    valid_payload[field] = value

    response = test_client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_rejects_missing_model(client, tmp_path, monkeypatch):
    import app.main as main

    test_client, _ = client
    app.dependency_overrides.pop(get_model_bundle, None)

    missing_path = tmp_path / "missing_model.joblib"
    monkeypatch.setattr(main, "MODEL_PATH", missing_path)
    main.load_model_bundle.cache_clear()

    try:
        response = test_client.get("/health")
    finally:
        main.load_model_bundle.cache_clear()

    assert response.status_code == 503
    assert str(missing_path) in response.json()["detail"]


def test_loads_dictionary_artifact_in_its_declared_feature_order(
    tmp_path, monkeypatch
):
    import app.main as main
    import joblib

    feature_columns = list(EXPECTED_FEATURE_COLUMNS)
    model = StubModel()
    model_path = tmp_path / "predictive_maintenance_model.joblib"
    joblib.dump({"model": model, "feature_columns": feature_columns}, model_path)
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    load_model_bundle.cache_clear()

    try:
        bundle = load_model_bundle()
    finally:
        load_model_bundle.cache_clear()

    assert bundle.model is not None
    assert bundle.feature_columns == tuple(feature_columns)
