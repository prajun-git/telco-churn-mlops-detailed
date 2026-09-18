import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import shap
from flask import Flask, request, jsonify

from configs.config import MODEL_PATH, ENCODERS_PATH

app = Flask(__name__)

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "predictions.jsonl")


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


# Load model + encoders once, at startup — not per-request (expensive to reload every time)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)

# SHAP explainer — must be created AFTER model is loaded
explainer = shap.TreeExplainer(model)


def preprocess(payload):
    """Convert incoming JSON into an encoded row the model can use."""
    df = pd.DataFrame([payload])

    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    # Column order must match training exactly
    expected_cols = list(model.feature_names_in_)
    df = df[expected_cols]
    return df


def explain_prediction(X, top_n=3):
    """Return the top N features driving this specific prediction."""
    shap_values = explainer.shap_values(X)
    churn_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    feature_impacts = list(zip(X.columns, churn_shap))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    top_features = feature_impacts[:top_n]
    return [
        {"feature": f, "impact": round(float(v), 4), "direction": "increases risk" if v > 0 else "decreases risk"}
        for f, v in top_features
    ]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()

    try:
        X = preprocess(payload)
    except Exception as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    explanation = explain_prediction(X)

    log_prediction(payload, prediction, probability)

    return jsonify({
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4),
        "top_factors": explanation
    }), 200


if __name__ == "__main__":
    from waitress import serve
    print("Starting server on http://0.0.0.0:5000")
    serve(app, host="0.0.0.0", port=5000)