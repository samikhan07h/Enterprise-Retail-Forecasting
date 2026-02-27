import mlflow

def log_experiment(model_name, metrics):

    mlflow.set_experiment("Enterprise_Retail_Forecasting")

    with mlflow.start_run(run_name=model_name):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)