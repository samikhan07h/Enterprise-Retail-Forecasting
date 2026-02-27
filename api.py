from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Retail Forecasting API")

MODEL_PATH = "models/best_model.pkl"

class ForecastRequest(BaseModel):
    sales: list[float]

@app.get("/")
def root():
    return {"message": "Retail Forecasting API is running"}

@app.post("/predict")
def predict(request: ForecastRequest):
    try:
        if not os.path.exists(MODEL_PATH):
            return {"error": "Model file not found. Train and save model first."}

        model = joblib.load(MODEL_PATH)

        data = pd.DataFrame({"sales": request.sales})
        prediction = model.predict(data)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}