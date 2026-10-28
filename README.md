# Predictive Maintenance API

FastAPI serving the Predictive Maintenance model trained in Google Colab.

## Model artifact

Copy the trained artifact to:

```text
models/predictive_maintenance_model.joblib
```

The file must be a joblib-serialized dictionary with `model` and
`feature_columns` keys. `feature_columns` must contain exactly these feature
names (the artifact's order is used when creating model input):

- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Type_H`
- `Type_L`
- `Type_M`

The artifact is excluded from Git because trained models can be large. Place it
in the `models/` directory before building a Docker image for deployment. If
the artifact is absent, `/health` and `/predict` return HTTP 503 with its
expected path.

Use the same scikit-learn version used to train the model where possible;
scikit-learn does not guarantee compatibility of serialized estimators across
versions.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

`GET /health` reports API/model readiness. `POST /predict` accepts all eight
feature names as JSON and returns `Target` (`0` or `1`) and its label
(`No Failure` or `Failure`).

## Test and build

```bash
python -m pytest -v
docker build -t predictive-maintenance-api .
```

Tests use a small in-memory model double and do not require the trained artifact.
GitHub Actions runs the tests and builds the Docker image on pushes and pull
requests targeting `main`.
