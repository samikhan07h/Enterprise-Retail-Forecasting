🚀 Enterprise Retail Forecasting Platform

An end-to-end enterprise-grade retail demand forecasting system with multi-model benchmarking, experiment tracking, explainability, and production deployment capabilities.

Built with a full ML engineering mindset: from data ingestion to model deployment.

📌 Project Overview

This platform forecasts retail demand using multiple modeling paradigms:

📈 Statistical Models (SARIMA)

🔮 Additive Time Series Model (Prophet)

🌲 Gradient Boosting (XGBoost)

🧠 Deep Learning (LSTM)

🧮 Ensemble Modeling

📊 Time-Series Cross Validation

🧠 SHAP Explainability

📦 MLflow Experiment Tracking

⚡ FastAPI Production API

🐳 Docker Containerization

🗄 SQL + CSV Data Ingestion

📊 Streamlit Business Dashboard

🏗 System Architecture
Data Source (CSV / SQL)
        ↓
Data Preprocessing
        ↓
Feature Engineering (Lag, Rolling, Date Features)
        ↓
Model Training
  ├── SARIMA
  ├── Prophet
  ├── XGBoost
  ├── LSTM
  └── Ensemble
        ↓
Time-Series Cross Validation
        ↓
Evaluation (MAE, RMSE, MAPE, CV_RMSE)
        ↓
MLflow Tracking
        ↓
Model Selection
        ↓
API Deployment (FastAPI)
        ↓
Streamlit Dashboard
📊 Model Benchmark Results
Model	MAE	RMSE	MAPE (%)	CV_RMSE
Prophet	1486.77	1776.32	5.62	—
XGBoost	4426.74	5636.58	14.77	—
SARIMA	22992.37	25475.16	74.92	39431.57
LSTM	12152.29	13896.61	38.16	—
Ensemble	2573.11	3117.26	8.21	—

🏆 Best Model (Lowest RMSE): Prophet

🧠 Explainability

SHAP (SHapley Additive exPlanations) is integrated for XGBoost to:

Identify feature importance

Understand lag influence

Explain demand drivers

Improve model transparency

📈 Cross Validation

Implemented time-series cross-validation (rolling window split):

Preserves temporal order

Prevents data leakage

Provides robust CV_RMSE evaluation

🗄 Data Sources

Supports:

CSV upload (Kaggle Retail Dataset format)

SQL database connection

Expected schema:

date, sales
📂 Project Structure
Enterprise Retail Forecasting/
│
├── app.py                # Streamlit dashboard
├── pipeline.py           # Main ML pipeline
├── api.py                # FastAPI production API
├── database.py           # SQL connection handler
├── mlflow_utils.py       # MLflow tracking logic
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
│
├── models/               # Saved models
├── utils/                # Helper utilities
└── mlruns/               # MLflow tracking logs
⚙ Installation
1️⃣ Clone Repository
git clone https://github.com/your-username/enterprise-retail-forecasting.git
cd enterprise-retail-forecasting
2️⃣ Create Virtual Environment
conda create -n retail_env python=3.10
conda activate retail_env
3️⃣ Install Dependencies
pip install -r requirements.txt
▶ Run Streamlit Dashboard
streamlit run app.py

Access at:

http://localhost:8501
🚀 Run FastAPI Production Server
uvicorn api:app --reload

Access API docs at:

http://localhost:8000/docs
🐳 Docker Deployment

Build container:

docker build -t retail-forecasting .

Run container:

docker run -p 8501:8501 retail-forecasting

Or using docker-compose:

docker-compose up --build
📊 MLflow Experiment Tracking

Start MLflow UI:

mlflow ui

Access at:

http://localhost:5000

Tracks:

Parameters

Metrics

Model versions

Artifacts

📈 Key Features

✔ Multi-model benchmarking
✔ Automated best model selection
✔ Cross-validation scoring
✔ SHAP explainability
✔ LSTM deep learning integration
✔ Ensemble forecasting
✔ API deployment ready
✔ Dockerized production setup
✔ MLflow experiment tracking
✔ SQL data ingestion

🎯 Business Impact

Detect seasonality and trend

Reduce inventory overstock

Improve demand planning

Compare model reliability

Enable scalable forecasting infrastructure

🧪 Future Improvements

Hyperparameter tuning (Optuna)

CI/CD integration

Cloud deployment (AWS/GCP/Azure)

Real-time streaming forecasts

Model registry versioning

👨‍💻 Author

Md Sami Ahmad
B.Tech CSE (Data Science & ML)
Focused on building production-grade ML systems.