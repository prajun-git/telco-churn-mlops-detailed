import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
from flask import Flask, request, jsonify

from configs.config import MODEL_PATH, ENCODERS_PATH

app = Flask(__name__)

# Load model + encoders once, at startup — not per-request (expensive to reload every time)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)


def preprocess(payload):
    """Convert incoming JSON into an encoded row the model can use."""
    df = pd.DataFrame([payload])

    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    # Column order must match training exactly
    expected_cols = list(model.feature_names_in_)
    df = df[expected_cols]
    return df


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

    return jsonify({
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)