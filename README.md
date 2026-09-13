# 🚀 Enterprise Retail Forecasting Platform

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production_API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **end-to-end retail demand forecasting platform** with multi-model benchmarking, experiment tracking, SHAP explainability, and a FastAPI serving layer — built to demonstrate real ML engineering practice, not just notebook-level modeling.

---

## 📌 Overview

The platform trains five forecasting approaches on a single sales time series, benchmarks them on a held-out split, then **refits the winning model on the full dataset** before persisting it for serving:

| Approach | Model | Notes |
|---|---|---|
| 📈 Statistical | SARIMA | `order=(1,1,1)`, weekly seasonality (`s=7`) — assumes **daily-frequency** data |
| 🔮 Additive decomposition | Prophet | Weekly seasonality always on; yearly seasonality only enabled with ≥365 data points |
| 🌲 Gradient Boosting | XGBoost | 14-lag sliding window regression (see [Feature Engineering](#-feature-engineering) below) |
| 🧠 Deep Learning | LSTM | 14-step lookback window, single LSTM(50) layer, `MinMaxScaler`-normalized |
| 🧮 Ensemble | Weighted blend | Inverse-RMSE weighted average of the four models above |

---

## 🏗️ System Architecture

```
Data Source (CSV / SQL)
        │
        ▼
Data Preprocessing (groupby date, sum sales)
        │
        ▼
Train/Test Split (80/20, min. 30 data points)
        │
        ▼
Model Training
   ├── SARIMA
   ├── Prophet
   ├── XGBoost
   ├── LSTM
   └── Ensemble
        │
        ▼
Evaluation (MAE, RMSE, MAPE) + Cross-Validation (SARIMA only)
        │
        ▼
Best Model Selection (lowest RMSE)
        │
        ▼
Refit Winner on FULL Series ──▶ Persist to models/forecast_artifact.pkl
        │
        ▼
MLflow Logging
        │
        ▼
Streamlit Dashboard ◀──────────────▶ FastAPI (/predict)
```

---

## 🧮 Feature Engineering

> **Note:** earlier drafts of this README described "Lag, Rolling, and Date" features. The actual implementation is **lag-only** — this section reflects the real code.

XGBoost is trained on a pure sliding-window representation: for a lag of 14, each training row is the previous 14 sales values, with the 15th as the target. Forecasting beyond the training data is done **recursively** — each prediction is fed back into the window to produce the next one, which means small errors can compound over longer horizons.

LSTM uses the same 14-step lookback idea, but on `MinMaxScaler`-normalized data through a single-layer LSTM network.

**Not currently implemented:** rolling statistics (e.g. 7-day rolling mean) and calendar/date features (day-of-week, month, holiday flags). Adding these to `models/xgb_model.py`'s `create_features()` is a natural next step and would likely improve XGBoost's accuracy — see [Future Improvements](#-future-improvements).

---

## 📊 Model Benchmark Results

| Model     | MAE       | RMSE      | MAPE (%) | CV_RMSE   |
|-----------|-----------|-----------|----------|-----------|
| **Prophet** 🏆 | 1,486.77  | 1,776.32  | 5.62     | —         |
| XGBoost   | 4,426.74  | 5,636.58  | 14.77    | —         |
| Ensemble  | 2,573.11  | 3,117.26  | 8.21     | —         |
| LSTM      | 12,152.29 | 13,896.61 | 38.16    | —         |
| SARIMA    | 22,992.37 | 25,475.16 | 74.92    | 39,431.57 |

> **Best Model (lowest RMSE): Prophet.** Results will vary by dataset — re-run the pipeline on your own data to get benchmark numbers specific to it.
>
> *CV_RMSE is currently only computed for SARIMA via rolling-window cross-validation. Extending this to all models is tracked in [Future Improvements](#-future-improvements).*

---

## 🧠 Explainability

[SHAP](https://shap.readthedocs.io/) is integrated for XGBoost only (the other model types don't expose a comparable per-prediction feature attribution in this implementation):

- Per-lag feature importance (`lag_14` … `lag_1`)
- SHAP summary plot over a 100-row sample of the training window
- Both are skipped automatically if XGBoost isn't the winning model for a given run

---

## 📈 Cross-Validation Strategy

Time-series cross-validation uses a rolling-window split for SARIMA:

- Preserves temporal order (no shuffling)
- Prevents data leakage from future to past
- Produces a `CV_RMSE` metric alongside the standard holdout RMSE

---

## 🗄️ Data Requirements

| Requirement | Value |
|---|---|
| Minimum data points | **30** (enforced — the pipeline raises a clear error below this) |
| Assumed frequency | **Daily.** Weekly/monthly data will run without erroring, but SARIMA's seasonal period (7) and Prophet's future-dataframe frequency won't align correctly with the actual cadence. |
| Required columns | A date column and a numeric sales/target column (auto-detected by name, with a manual override in the dashboard if detection fails) |
| Ingestion modes | CSV upload, or SQL query via `database.py` |

---

## 📂 Project Structure

```
Enterprise-Retail-Forecasting/
│
├── app.py                     # Streamlit dashboard
├── pipeline.py                 # ForecastPipeline + forecast_from_artifact()
├── api.py                      # FastAPI serving layer
├── database.py                  # SQL connection handler
├── mlflow_utils.py               # MLflow tracking logic
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
│
├── models/
│   ├── sarima_model.py
│   ├── prophet_model.py
│   ├── xgb_model.py            # create_features, DEFAULT_LAG
│   ├── lstm_model.py            # forecast_with_lstm, DEFAULT_LOOK_BACK
│   ├── forecast_artifact.pkl    # generated at runtime — the winning model + data
│   ├── lstm_model.keras          # generated only if LSTM wins
│   └── lstm_scaler.pkl           # generated only if LSTM wins
│
├── utils/
│   ├── cross_validation.py
│   ├── metrics.py
│   └── shap_explainer.py
│
└── mlruns/                     # MLflow tracking logs
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/samikhan07h/Enterprise-Retail-Forecasting.git
cd Enterprise-Retail-Forecasting
```

### 2️⃣ Create a virtual environment
```bash
conda create -n retail_env python=3.10
conda activate retail_env
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure secrets
```bash
cp .env.example .env
# For Streamlit specifically:
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Fill in real values in the copies — **never commit `.env` or `secrets.toml`**.

---

## ▶️ Running the Platform

### Streamlit Dashboard
```bash
streamlit run app.py
```
→ [http://localhost:8501](http://localhost:8501)

### FastAPI Serving Layer
```bash
uvicorn api:app --reload
```
→ Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)

**API contract:** `POST /predict` takes `{"horizon": <int, 1-90>}` and returns `{"model": <winning model name>, "predictions": [...]}`. It does **not** accept live sales data — it forecasts forward from wherever the last dashboard run's training series ended. Re-run the dashboard's pipeline to refresh the underlying artifact. `GET /health` reports whether a trained artifact is currently available.

### MLflow Experiment Tracking
```bash
mlflow ui
```
→ [http://localhost:5000](http://localhost:5000)

---

## 🐳 Docker Deployment

```bash
docker build -t retail-forecasting .
docker run -p 8501:8501 retail-forecasting
# or
docker-compose up --build
```

---

## 🔐 Secrets & Configuration

- **Local development:** `.env` (Docker/API) and `.streamlit/secrets.toml` (dashboard) — both git-ignored
- **Streamlit Cloud:** App settings → Secrets, TOML format:
  ```toml
  [database]
  connection_string = "postgresql://user:password@host:port/dbname"
  ```
- **Docker/production:** inject via environment variables or a secrets manager (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault, etc.)

> ⚠️ Never commit real credentials. Rotate immediately if one is accidentally pushed.

---

## ⚠️ Known Limitations

- **Daily-frequency assumption** in SARIMA and Prophet (see [Data Requirements](#-data-requirements))
- **Feature engineering is lag-only** for XGBoost — no rolling stats or calendar features yet
- **Recursive multi-step forecasting** (XGBoost, LSTM) compounds error over longer horizons; accuracy degrades the further out you forecast
- **SHAP explainability is XGBoost-only** — no comparable explanation is generated for the other model types
- **Cross-validation is SARIMA-only** — other models are evaluated on a single holdout split
- **The API serves one artifact at a time** — it reflects whichever model won the most recent dashboard run, not a per-request choice

---

## 📈 Key Features

- ✔ Multi-model benchmarking across statistical, ML, and DL approaches
- ✔ Automatic refit-on-full-data before persisting the winning model
- ✔ SHAP explainability for XGBoost
- ✔ FastAPI serving layer with a documented, versioned contract
- ✔ Dockerized, reproducible environment
- ✔ MLflow experiment tracking
- ✔ Flexible SQL + CSV ingestion with secrets-based DB credentials

---

## 🧪 Future Improvements

- [ ] Add rolling-window and calendar (day-of-week, month, holiday) features to XGBoost
- [ ] Extend cross-validation to all models, not just SARIMA
- [ ] Auto-detect or let users specify data frequency instead of assuming daily
- [ ] Hyperparameter tuning with Optuna
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Cloud deployment (AWS / GCP / Azure)
- [ ] MLflow Model Registry integration
- [ ] Unit test coverage for `pipeline.py` and each `models/*.py`

---

## 👨‍💻 Author

**Md Sami Ahmad**
B.Tech CSE (Data Science & ML)

[GitHub](https://github.com/samikhan07h)

---

## 📄 License

MIT — see [LICENSE](LICENSE).
