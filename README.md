# 🚀 Enterprise Retail Forecasting Platform

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production_API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **end-to-end, production-grade retail demand forecasting platform** featuring multi-model benchmarking, experiment tracking, explainability, and deployment-ready infrastructure.

Built with a full ML engineering mindset — from raw data ingestion to a live, monitored API.

---

## 📌 Overview

This platform forecasts retail demand using multiple complementary modeling paradigms, then automatically selects the best-performing model based on cross-validated error metrics.

| Approach | Model | Why it's included |
|---|---|---|
| 📈 Statistical | SARIMA | Captures seasonality & trend in classical time series |
| 🔮 Additive decomposition | Prophet | Robust to missing data & holiday effects |
| 🌲 Gradient Boosting | XGBoost | Captures non-linear feature interactions |
| 🧠 Deep Learning | LSTM | Learns long-range temporal dependencies |
| 🧮 Ensemble | Weighted blend | Combines model strengths, reduces variance |

---

## 🏗️ System Architecture

```
Data Source (CSV / SQL)
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering (Lag, Rolling, Date Features)
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
Time-Series Cross-Validation
        │
        ▼
Evaluation (MAE, RMSE, MAPE, CV_RMSE)
        │
        ▼
MLflow Tracking
        │
        ▼
Model Selection
        │
        ▼
API Deployment (FastAPI) ──▶ Streamlit Dashboard
```

---

## 📊 Model Benchmark Results

| Model     | MAE       | RMSE      | MAPE (%) | CV_RMSE   |
|-----------|-----------|-----------|----------|-----------|
| **Prophet** 🏆 | 1,486.77  | 1,776.32  | 5.62     | —         |
| XGBoost   | 4,426.74  | 5,636.58  | 14.77    | —         |
| Ensemble  | 2,573.11  | 3,117.26  | 8.21     | —         |
| LSTM      | 12,152.29 | 13,896.61 | 38.16    | —         |
| SARIMA    | 22,992.37 | 25,475.16 | 74.92    | 39,431.57 |

> **Best Model (lowest RMSE): Prophet**
>
> *Note: CV_RMSE is currently only computed for SARIMA. Extending rolling-window cross-validation to all models is tracked in [Future Improvements](#-future-improvements).*

---

## 🧠 Explainability

[SHAP](https://shap.readthedocs.io/) (SHapley Additive exPlanations) is integrated for the XGBoost model to:

- Identify global and per-prediction feature importance
- Quantify the influence of lag and rolling-window features
- Explain key demand drivers to business stakeholders
- Improve overall model transparency and trust

---

## 📈 Cross-Validation Strategy

Time-series cross-validation uses a **rolling-window split** to ensure realistic evaluation:

- Preserves temporal order (no shuffling)
- Prevents data leakage from future to past
- Produces a robust, generalizable CV_RMSE metric

---

## 🗄️ Data Sources

The platform supports two ingestion modes:

- **CSV upload** — compatible with the Kaggle Retail Dataset format
- **SQL database connection** — via `database.py`

**Expected schema:**

| Column | Type | Description |
|---|---|---|
| `date`  | datetime | Observation date |
| `sales` | float    | Target variable to forecast |

---

## 📂 Project Structure

```
Enterprise-Retail-Forecasting/
│
├── app.py                # Streamlit dashboard
├── pipeline.py            # Main ML pipeline
├── api.py                 # FastAPI production API
├── database.py             # SQL connection handler
├── mlflow_utils.py          # MLflow tracking logic
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
│
├── models/                # Saved model artifacts
├── utils/                  # Helper utilities
└── mlruns/                 # MLflow tracking logs
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
Copy the example env file and fill in your own credentials — **never commit real secrets**:
```bash
cp .env.example .env
```

---

## ▶️ Running the Platform

### Streamlit Dashboard
```bash
streamlit run app.py
```
Access at → [http://localhost:8501](http://localhost:8501)

### FastAPI Production Server
```bash
uvicorn api:app --reload
```
Access interactive API docs at → [http://localhost:8000/docs](http://localhost:8000/docs)

### MLflow Experiment Tracking
```bash
mlflow ui
```
Access at → [http://localhost:5000](http://localhost:5000)

Tracks: parameters, metrics, model versions, and artifacts.

---

## 🐳 Docker Deployment

**Build the image:**
```bash
docker build -t retail-forecasting .
```

**Run the container:**
```bash
docker run -p 8501:8501 retail-forecasting
```

**Or use Docker Compose:**
```bash
docker-compose up --build
```

---

## 🔐 Secrets & Configuration

This project reads sensitive configuration (DB credentials, API tokens, etc.) from environment variables rather than hard-coded values.

- **Local development:** use a `.env` file (excluded via `.gitignore`)
- **Streamlit Cloud:** use the built-in **Secrets manager** (`App settings → Secrets`), written in TOML:
  ```toml
  DB_USERNAME = "myuser"
  DB_TOKEN = "your-token-here"

  [some_section]
  some_key = 1234
  ```
- **Docker/production:** inject secrets via environment variables or a secrets manager (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault, etc.)

> ⚠️ Never commit real credentials to the repository. Rotate any key immediately if it is accidentally pushed to version control.

---

## 📈 Key Features

- ✔ Multi-model benchmarking across statistical, ML, and DL approaches
- ✔ Automated best-model selection based on RMSE
- ✔ Rolling-window cross-validation scoring
- ✔ SHAP-based explainability
- ✔ LSTM deep learning integration
- ✔ Ensemble forecasting
- ✔ Production-ready FastAPI deployment
- ✔ Dockerized, reproducible environment
- ✔ MLflow experiment tracking
- ✔ Flexible SQL + CSV data ingestion

---

## 🎯 Business Impact

- Detect seasonality and demand trends early
- Reduce inventory overstock and stockouts
- Improve demand planning accuracy
- Compare model reliability objectively before deployment
- Enable scalable, repeatable forecasting infrastructure

---

## 🧪 Future Improvements

- [ ] Extend rolling-window cross-validation to all models (not just SARIMA)
- [ ] Hyperparameter tuning with Optuna
- [ ] CI/CD pipeline (GitHub Actions) for automated testing & deployment
- [ ] Cloud deployment (AWS / GCP / Azure)
- [ ] Real-time streaming forecasts
- [ ] Model registry & versioning (MLflow Model Registry)
- [ ] Unit & integration test coverage

---

## 👨‍💻 Author

**Md Sami Ahmad**
B.Tech CSE (Data Science & ML)
Focused on building production-grade ML systems.

[GitHub](https://github.com/samikhan07h) · [LinkedIn](#) · [Portfolio](#)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

### ⭐ If you find this project useful, consider giving it a star on GitHub!
