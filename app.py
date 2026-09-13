"""
Enterprise Retail Forecasting Platform
---------------------------------------
A production-style Streamlit dashboard for multi-model retail demand
forecasting (SARIMA, Prophet, XGBoost, LSTM, Ensemble) with SHAP
explainability and SQL/CSV ingestion.

Author: Md Sami Ahmad
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pipeline import ForecastPipeline
from database import load_from_sql

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Retail Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Custom styling (pairs with the dark theme set in .streamlit/config.toml)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1c1f26;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Retail Forecasting")
    st.caption("Enterprise-grade demand forecasting platform")
    st.divider()
    st.markdown(
        """
        **Models included**
        - 📈 SARIMA
        - 🔮 Prophet
        - 🌲 XGBoost
        - 🧠 LSTM
        - 🧮 Weighted Ensemble
        """
    )
    st.divider()
    st.markdown(
        "Built by **Md Sami Ahmad**  \n"
        "[GitHub](https://github.com/samikhan07h) · B.Tech CSE (Data Science & ML)"
    )

st.title("🚀 Enterprise Retail Forecasting Platform")
st.caption(
    "Upload sales data or connect a database, benchmark five forecasting "
    "models, and explore SHAP-based explainability — all in one dashboard."
)

# ----------------------------------------------------------------------------
# Data source selection
# ----------------------------------------------------------------------------
tab_csv, tab_sql = st.tabs(["📁 CSV Upload", "🗄️ SQL Database"])

df = None

with tab_csv:
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    st.caption("Expected columns: a date column and a sales/target column.")
    if file:
        df = pd.read_csv(file)

with tab_sql:
    # Prefer a connection string stored in Streamlit Secrets over typing it in.
    default_conn = ""
    if "database" in st.secrets:
        default_conn = st.secrets["database"].get("connection_string", "")

    conn = st.text_input(
        "SQL Connection String",
        value=default_conn,
        type="password",
        help=(
            "e.g. postgresql://user:password@host:port/dbname . "
            "On Streamlit Cloud, store this in App settings → Secrets "
            "under [database] connection_string instead of typing it here."
        ),
    )
    table = st.text_input("Table Name")

    if st.button("Load from database", disabled=not (conn and table)):
        try:
            with st.spinner("Querying database..."):
                df = load_from_sql(conn, table)
        except Exception as e:
            st.error(f"Could not load data: {e}")

# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
if df is not None and not df.empty:

    with st.expander("🔍 Preview raw data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    # Auto-detect columns, but always let the user override
    date_cols = [c for c in df.columns if "date" in c.lower()]
    sales_cols = [c for c in df.columns if "sales" in c.lower()]

    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox(
            "Date column",
            options=list(df.columns),
            index=df.columns.get_loc(date_cols[0]) if date_cols else 0,
        )
    with col2:
        target_col = st.selectbox(
            "Target (sales) column",
            options=list(df.columns),
            index=df.columns.get_loc(sales_cols[0]) if sales_cols else 0,
        )

    run = st.button("▶️ Run Forecast Pipeline", type="primary")

    if run:
        try:
            work_df = df.copy()
            work_df[date_col] = pd.to_datetime(work_df[date_col])
            work_df = work_df.sort_values(by=date_col)

            with st.spinner("Training models — this can take a minute..."):
                pipeline = ForecastPipeline(work_df, date_col, target_col)
                pipeline.train_test_split()
                pipeline.run_models()
                results = pipeline.evaluate_models()

            st.session_state["pipeline"] = pipeline
            st.session_state["results"] = pd.DataFrame(results).T
            st.success("Pipeline finished successfully.")

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    # ------------------------------------------------------------------
    # Results (persist across reruns via session_state)
    # ------------------------------------------------------------------
    if "results" in st.session_state:
        pipeline = st.session_state["pipeline"]
        results_df = st.session_state["results"]

        st.subheader("📊 Model Benchmark")

        highlight_cols = [c for c in ["MAE", "RMSE", "MAPE"] if c in results_df.columns]
        st.dataframe(
            results_df.style.highlight_min(subset=highlight_cols, color="#1f8f5f"),
            use_container_width=True,
        )

        if "RMSE" not in results_df.columns:
            st.warning("RMSE column missing from results.")
            st.stop()

        best_model = results_df["RMSE"].idxmin()

        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Best Model", best_model)
        c2.metric("RMSE", f"{results_df.loc[best_model, 'RMSE']:.2f}")
        if "MAPE" in results_df.columns:
            c3.metric("MAPE", f"{results_df.loc[best_model, 'MAPE']:.2f}%")

        st.download_button(
            "⬇️ Download benchmark results (CSV)",
            results_df.to_csv().encode("utf-8"),
            file_name="benchmark_results.csv",
            mime="text/csv",
        )

        # ------------------------------------------------------------
        # Forecast vs Actual (interactive)
        # ------------------------------------------------------------
        st.subheader("📈 Forecast vs Actual")
        ensemble_pred = pipeline.weighted_ensemble()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pipeline.test_dates, y=pipeline.test,
            mode="lines", name="Actual", line=dict(color="#4C9AFF"),
        ))
        fig.add_trace(go.Scatter(
            x=pipeline.test_dates, y=ensemble_pred,
            mode="lines", name="Ensemble Forecast",
            line=dict(color="#FF7A59", dash="dash"),
        ))
        fig.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------
        # Explainability (XGBoost only)
        # ------------------------------------------------------------
        if best_model == "XGBoost":
            st.subheader("🔎 XGBoost Explainability")
            imp_col, shap_col = st.columns(2)

            from models.xgb_model import create_features, DEFAULT_LAG
            feature_names = [f"lag_{DEFAULT_LAG - i}" for i in range(DEFAULT_LAG)]

            with imp_col:
                st.caption("Feature importance")
                importance = pd.Series(
                    pipeline.xgb_model.feature_importances_, index=feature_names
                )
                st.bar_chart(importance)

            with shap_col:
                st.caption("SHAP summary")
                try:
                    import shap

                    X_sample, _ = create_features(pipeline.train)
                    X_sample = pd.DataFrame(X_sample[:100], columns=feature_names)

                    explainer = shap.TreeExplainer(pipeline.xgb_model)
                    shap_values = explainer.shap_values(X_sample)

                    plt.style.use("dark_background")
                    shap_fig = plt.figure()
                    shap.summary_plot(shap_values, X_sample, show=False)
                    st.pyplot(shap_fig, use_container_width=True)
                except Exception as e:
                    st.info(f"SHAP explanation unavailable: {e}")

else:
    st.info("👆 Upload a CSV or connect to a database to get started.")
