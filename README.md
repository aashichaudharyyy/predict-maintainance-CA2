Predictive Maintenance API

A machine learning-powered REST API that predicts potential machine failures using operational parameters. This project combines a trained Random Forest classifier with FastAPI, automated testing, Docker, and GitHub Actions CI.

Project Resources

Google Colab Notebook: View Training Notebook

Dataset: Machine Predictive Maintenance Classification — Kaggle

GitHub Repository: predict-maintainance-CA2

Overview

Predictive maintenance uses machine learning to identify potential equipment failures from operating conditions. Instead of relying only on scheduled maintenance, such predictions can help flag machines that may require inspection.

This project provides a REST API that accepts machine operating parameters and returns a binary machine-failure prediction.

Dataset

The project uses the Machine Predictive Maintenance Classification dataset available on Kaggle.

Dataset details:

Source: Kaggle

Dataset author: shivamb

File: predictive_maintenance.csv

Records: 10,000

Original columns: 10

The dataset includes machine operating conditions and failure-related information.

Features used by the model

Air temperature

Process temperature

Rotational speed

Torque

Tool wear

Machine type, represented through one-hot encoded features

The target variable is Target, which indicates whether a machine failure occurred.

Dataset source: Kaggle — Machine Predictive Maintenance Classification

Model Training

The model training workflow is documented in the linked Google Colab notebook.

The trained machine learning model is serialized using Joblib and included in the repository:

models/predictive_maintenance_model.joblib

The FastAPI application loads this artifact to generate predictions. The input features and preprocessing used by the API must match the model's training configuration.

Technology Stack

Python — application development

pandas — data handling

scikit-learn — machine learning model

Joblib — model serialization

FastAPI — REST API

Pydantic — request validation

pytest — automated testing

Docker — containerization

GitHub Actions — continuous integration

API Endpoints

Method

Endpoint

Description

GET

/health

Checks API health

POST

/predict

Predicts machine failure

Interactive API documentation is available at /docs when the application is running.

Prediction Output

The prediction endpoint returns:

Target: binary prediction (0 or 1)

Prediction: human-readable label (No Failure or Failure)

The result is a model prediction and should not be treated as a guarantee of future machine behaviour.

Continuous Integration

The GitHub Actions workflow is defined in .github/workflows/ci.yml.

On pushes and pull requests targeting main, the workflow:

Sets up Python.

Installs project dependencies.

Runs automated API tests using pytest.

Builds the Docker image.

This provides an automated check that the tests pass and the Docker image can be built.
