import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pipeline import ForecastPipeline
from database import load_from_sql

st.set_page_config(layout="wide")
st.title("🚀 Enterprise Retail Forecasting Platform")

# -----------------------------
# Data Source Selection
# -----------------------------

source = st.radio("Select Data Source", ["CSV Upload", "SQL Database"])

df = None

if source == "CSV Upload":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

elif source == "SQL Database":
    conn = st.text_input("SQL Connection String")
    table = st.text_input("Table Name")
    if conn and table:
        df = load_from_sql(conn, table)

# -----------------------------
# Main Pipeline Execution
# -----------------------------

if df is not None:

    # Auto-detect columns
    date_cols = [c for c in df.columns if "date" in c.lower()]
    sales_cols = [c for c in df.columns if "sales" in c.lower()]

    if not date_cols or not sales_cols:
        st.error("Could not automatically detect date or sales column.")
        st.stop()

    date_col = date_cols[0]
    target_col = sales_cols[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(by=date_col, inplace=True)

    # -----------------------------
    # Run Forecast Pipeline
    # -----------------------------

    pipeline = ForecastPipeline(df, date_col, target_col)
    pipeline.train_test_split()
    pipeline.run_models()

    results = pipeline.evaluate_models()
    results_df = pd.DataFrame(results).T

    # -----------------------------
    # Display Benchmark Results
    # -----------------------------

    st.subheader("📊 Benchmark Results")
    st.dataframe(results_df)

    # Determine Best Model
    if "RMSE" in results_df.columns:
        best_model = results_df["RMSE"].idxmin()
        st.success(f"🏆 Best Model: {best_model}")
    else:
        st.warning("RMSE column missing.")
        st.stop()

    # -----------------------------
    # Show Ensemble Forecast
    # -----------------------------

    ensemble_pred = pipeline.weighted_ensemble()

    st.subheader("📈 Forecast vs Actual")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pipeline.test_dates, pipeline.test, label="Actual")
    ax.plot(pipeline.test_dates, ensemble_pred, label="Ensemble")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------
    # XGBoost Explainability
    # -----------------------------

    if best_model == "XGBoost":

        st.subheader("🔎 XGBoost Feature Importance")

        st.bar_chart(pipeline.xgb_model.feature_importances_)

        try:
            import shap
            from models.xgb_model import create_features

            st.subheader("🧠 SHAP Summary Plot")

            X_sample, _ = create_features(pipeline.train)
            X_sample = X_sample[:100]

            explainer = shap.TreeExplainer(pipeline.xgb_model)
            shap_values = explainer.shap_values(X_sample)

            shap_fig = plt.figure()
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(shap_fig)

        except Exception as e:
            st.warning(f"SHAP could not be generated: {e}")
