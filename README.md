AQI Forecasting MLOps Platform

End-to-end AQI forecasting system for Karachi, Lahore, and Islamabad built during my Data Science internship at 10Pearls.

The project collects live air-quality and weather data, maintains historical data in BigQuery, builds time-series features, trains separate models for 24-hour, 48-hour, and 72-hour forecasting, tracks experiments with MLflow, and serves predictions through FastAPI and Streamlit.

Author: Syed Abdullah Bin Masood
Role: Data Science Intern

Contents

Overview

Architecture

What the system does

Data pipeline

Feature engineering

Training pipeline

Model results

MLflow

Prediction pipeline

FastAPI

Streamlit dashboard

Explainability

AQI alerts

Automation

Cloud authentication

Project structure

Run locally

Screenshots used in this README

Current limitations

Overview

This project was built around one requirement: make AQI forecasting work as a repeatable MLOps pipeline, not as a one-off notebook.

The system currently supports:

Karachi

Lahore

Islamabad

and produces three direct forecast points:

+24 hours

+48 hours

+72 hours

The production workflow includes:

AQICN / WAQI data ingestion

OpenWeather weather data

BigQuery historical storage

temporal and rolling feature engineering

leakage-aware train/validation/test preparation

multi-model training

MLflow experiment tracking

model packaging

FastAPI inference

Streamlit dashboard

SHAP / LIME explainability

hazardous AQI email alerts

hourly and daily GitHub Actions

Architecture

flowchart LR
    A["AQICN / WAQI<br/>OpenWeather"] --> B["Validation"]
    B --> C["Canonical Feature Pipeline"]
    C --> D["BigQuery"]

    D --> E["Training Dataset"]
    E --> F["24h / 48h / 72h Training"]
    F --> G["MLflow"]
    G --> H["Model Bundles"]

    D --> I["Recent City Context"]
    A --> I
    H --> I

    I --> J["Prediction Pipeline"]
    J --> K["FastAPI"]
    K --> L["Streamlit"]
    K --> M["AQI Alerts"]

The project uses the same feature logic during training and inference so the serving path does not silently drift away from the training pipeline.

What the system does

1. Collects live data

Air-quality observations are collected from AQICN / WAQI.

Weather context is collected from OpenWeather.

Representative inputs include:

AQI
PM2.5
PM10
NO2
SO2
CO
O3
temperature
humidity
pressure
visibility
wind speed
wind direction
cloudiness

2. Stores historical observations

Historical data is kept in Google BigQuery.

Current table:

aqi_feature_store.engineered_features

The physical BigQuery table contains 31 columns.

The hourly pipeline uses append-only writes and avoids inserting duplicate city/hour records.

3. Builds model features

The canonical feature pipeline creates:

temporal features

lag features

rolling-window statistics

trend features

interaction features

air-quality features

spatial/city features

scaled and encoded model inputs

The final model contract contains:

606 ordered features
594 scaled numerical features

4. Trains separate forecast horizons

Instead of generating one prediction and repeating it across the next three days, the project trains separate forecasting targets for:

AQI(t + 24h)
AQI(t + 48h)
AQI(t + 72h)

5. Serves the result

The selected model bundle is loaded by the prediction service and exposed through FastAPI.

Streamlit consumes the API and shows:

current AQI

current AQI category

weather context

historical trend

24h forecast

48h forecast

72h forecast

model evaluation

explanation factors

Data pipeline

flowchart LR
    A["Live API Data"] --> B["Normalize Schema"]
    B --> C["Validate"]
    C --> D["City / Hour Deduplication"]
    D --> E["BigQuery WRITE_APPEND"]

Important pipeline behavior:

incoming timestamps are normalized

city names are canonicalized

duplicate logical city/hour records are filtered

optional weather values can remain missing

missing optional values are not replaced with arbitrary constants

BigQuery evidence



Feature engineering

The feature pipeline runs in a fixed order:

flowchart LR
    A["Temporal"] --> B["Lag"]
    B --> C["Rolling"]
    C --> D["Trend"]
    D --> E["Interaction"]
    E --> F["Air Quality"]
    F --> G["Spatial"]
    G --> H["Scaling / Encoding"]

Feature groups in the current pipeline:

Group

Intermediate additions

Temporal

41

Lag

53

Rolling

385

Trend

91

Interaction

6

Air Quality

6

Spatial

4

Preprocessing statistics are fitted on the training split and reused for validation, test, and inference.

This includes:

imputation statistics

scaling parameters

categorical alignment

pollutant normalization

final feature order

Training pipeline

The prepared chronological dataset contains:

Split

Rows

Train

66,549

Validation

14,767

Test

14,660

The validation split is used for model selection.

The test split is kept for final evaluation after the winner has already been selected.

Models in the scheduled comparison

Ridge Regression

Random Forest

HistGradientBoosting

Additional standalone experimentation scripts include XGBoost, LSTM, Prophet, and a Ridge baseline.

Model-selection flow

flowchart LR
    A["Train Split"] --> B["Candidate Models"]
    B --> C["Validation RMSE"]
    C --> D["Select Champion"]
    D --> E["Held-out Test Evaluation"]
    E --> F["MLflow + Model Bundle"]

Model results

Latest verified champion models:

Horizon

Champion

Test RMSE

Test R²

24h

Ridge Regression

19.1054

0.8146

48h

HistGradientBoosting

24.0924

0.7129

72h

Ridge Regression

28.8290

0.5874

RMSE, MAE, and R² are logged during training.

I keep train, validation, and test metrics separate instead of mixing values from different splits in one result row.

MLflow

MLflow is used to track horizon-specific training runs.

Each run records:

model parameters

training metrics

validation metrics

test metrics

horizon

model artifact

run ID

training timestamp

The final model bundle contains the model together with the fitted transformer and ordered feature schema.

MLflow experiment



Prediction pipeline

The inference path uses both recent historical context and the latest available observation.

flowchart LR
    A["Recent City History"] --> C["Merge + Validate"]
    B["Latest Live Observation"] --> C
    C --> D["Canonical Features"]
    D --> E["Persisted Transformer"]
    E --> F["606-Feature Contract"]
    F --> G["24h / 48h / 72h Bundle"]
    G --> H["Forecast"]

The predictor does not create missing model features with zeroes.

If the feature contract is wrong, inference fails instead of silently producing a questionable prediction.

A verified direct inference run produced:

Horizon

Predicted AQI

+24h

73.23

+48h

115.24

+72h

62.11

FastAPI

The backend exposes the prediction and dashboard services.

Main endpoints:

GET /
GET /api/v1/health
GET /api/v1/dashboard
GET /api/v1/dashboard/{city}
GET /api/v1/dashboard/{city}/explain

API documentation



Streamlit dashboard

The Streamlit application presents the forecasting output for all three supported cities.

Karachi



Lahore



Islamabad



The frontend shows unavailable values as unavailable rather than converting them to 0.

Explainability

The explainability layer is intentionally separated from prediction.

If an explanation fails, the forecast remains available.

Ridge / linear models

For linear models, local contribution can be calculated from the transformed feature value and learned coefficient.

Tree models

Compatible tree models can use SHAP TreeExplainer.

LIME

LIME is used for local explanation around an individual prediction.

SHAP / model contribution



LIME



AQI alerts

Hazardous AQI conditions can trigger an email notification.

flowchart LR
    A["Current / Forecast AQI"] --> B["AQI Category"]
    B --> C{"Hazardous?"}
    C -- "Yes" --> D["Email Alert"]
    C -- "No" --> E["Normal Response"]

Email delivery is treated as a side effect.

A failed email does not change the prediction result.

Automation

Two GitHub Actions workflows automate the project.

Hourly feature pipeline

Schedule:

cron: "17 * * * *"

The workflow:

authenticates to Google Cloud

fetches live AQI/weather data

validates and canonicalizes the payload

builds the hourly feature output

checks existing BigQuery keys

appends new city/hour records

stores workflow logs



Daily training pipeline

Schedule:

cron: "30 1 * * *"

Approximately:

06:30 PKT

The workflow:

loads the historical training data

creates exact 24h / 48h / 72h targets

rebuilds the training features

trains model candidates

selects each champion using validation RMSE

evaluates the champion on held-out test data

saves model and MLflow artifacts



Cloud authentication

GitHub Actions authenticates to Google Cloud using OIDC and Workload Identity Federation.

flowchart LR
    A["GitHub Actions"] --> B["OIDC Token"]
    B --> C["Google Workload Identity Federation"]
    C --> D["Service Account"]
    D --> E["BigQuery"]

This avoids keeping a long-lived Google service-account private key in the repository.

Project structure

src/
├── api/
│   ├── app.py
│   ├── routes.py
│   └── schemas.py
│
├── alerts/
│   ├── aqi_alert_service.py
│   └── email_alert_service.py
│
├── configs/
│   └── settings.py
│
├── dashboard/
│   ├── streamlit_app.py
│   └── components.py
│
├── explainability/
│   ├── dashboard_explainer.py
│   ├── shap_analysis.py
│   └── lime_analysis.py
│
├── feature_engineering/
│   ├── pipeline_steps.py
│   ├── temporal_features.py
│   ├── lag_features.py
│   ├── rolling_features.py
│   ├── trend_features.py
│   ├── interaction_features.py
│   ├── air_quality_features.py
│   ├── spatial_features.py
│   ├── scaling_encoding.py
│   └── build_features.py
│
├── feature_pipeline/
│   └── run_pipeline.py
│
├── feature_store/
│   └── bigquery_feature_store.py
│
├── prediction/
│   ├── dashboard_service.py
│   ├── feature_pipeline.py
│   ├── forecast.py
│   ├── live_data_service.py
│   ├── load_features.py
│   ├── load_model.py
│   ├── predictor.py
│   └── validator.py
│
└── training/
    ├── forecast_targets.py
    ├── dataset.py
    ├── model_bundle.py
    ├── train_multi_models.py
    └── run_pipeline.py

.github/
└── workflows/
    ├── hourly_feature_pipeline.yml
    └── training_pipeline.yml

models/
└── registry/

data/
├── training/
└── predictions/

docs/
└── images/

Run locally

Run commands from the repository root.

Activate environment

.venv\Scripts\activate

FastAPI backend

python -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000

Swagger:

http://127.0.0.1:8000/docs

Streamlit frontend

streamlit run src/dashboard/streamlit_app.py

Feature pipeline

python -m src.feature_pipeline.run_pipeline

Training pipeline

python -m src.training.run_pipeline

Direct forecast

python -m src.prediction.forecast

Screenshots used in this README

Create:

docs/images/

and add these files:

bigquery_table.png
mlflow_runs.png
api_docs.png

dashboard_karachi.png
dashboard_lahore.png
dashboard_islamabad.png

shap_summary.png
lime_explanation.png

github_hourly_green.png
github_daily_green.png

The paths are already referenced in this README.

Current limitations

The current implementation covers the functional MLOps lifecycle required for the internship project.

Production hardening can still be improved in areas such as:

dependency locking

stronger automated test gates

partition-aware BigQuery training reads

centralized persistent monitoring

model promotion and rollback controls

deployment-specific API caching and scaling

The GitHub-hosted data and training workflows already run independently of the local machine.

FastAPI and Streamlit are kept as separate serving components and can be deployed to the runtime selected for the project.

Author

Syed Abdullah Bin Masood
Data Science Intern — 10Pearls
