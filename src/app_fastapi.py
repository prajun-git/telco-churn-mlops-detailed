import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from configs.config import MODEL_PATH, ENCODERS_PATH

app = FastAPI(title="Telco Churn Prediction API", version="1.0")

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "predictions.jsonl")

# Load model + encoders once, at startup
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)

explainer = shap.TreeExplainer(model)


# Pydantic model = automatic request validation.
# FastAPI rejects malformed requests (missing field, wrong type) before this code even runs.
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    top_factors: list


def log_prediction(payload, prediction, probability):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": payload,
        "prediction": "Yes" if prediction == 1 else "No",
        "probability": round(float(probability), 4),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def preprocess(payload: dict):
    df = pd.DataFrame([payload])
    for col, le in encoders.items():
        df[col] = le.transform(df[col])
    expected_cols = list(model.feature_names_in_)
    df = df[expected_cols]
    return df


def explain_prediction(X, top_n=3):
    shap_values = explainer.shap_values(X)
    churn_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    feature_impacts = list(zip(X.columns, churn_shap))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    return [
        {"feature": f, "impact": round(float(v), 4), "direction": "increases risk" if v > 0 else "decreases risk"}
        for f, v in feature_impacts[:top_n]
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    payload = customer.model_dump()

    try:
        X = preprocess(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    explanation = explain_prediction(X)

    log_prediction(payload, prediction, probability)

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4),
        "top_factors": explanation,
    }