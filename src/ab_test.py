import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from configs.config import MODEL_PATH, ENCODERS_PATH, VAL_PATH, MODEL_DIR


def load_model_and_encoders(model_path, encoders_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


def evaluate(model, encoders, data_path):
    df = pd.read_csv(data_path).drop(columns=["customerID"]).copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    for col, le in encoders.items():
        df[col] = le.transform(df[col])
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "f1": f1_score(y, y_pred),
        "auc": roc_auc_score(y, y_proba),
    }


def run(candidate_model_path, candidate_encoders_path):
    print("Loading champion (A) and candidate (B) models...")
    champion_model, champion_encoders = load_model_and_encoders(MODEL_PATH, ENCODERS_PATH)
    candidate_model, candidate_encoders = load_model_and_encoders(candidate_model_path, candidate_encoders_path)

    print("Evaluating both on validation data...\n")
    champion_metrics = evaluate(champion_model, champion_encoders, VAL_PATH)
    candidate_metrics = evaluate(candidate_model, candidate_encoders, VAL_PATH)

    print(f"{'Metric':<10} {'Champion (A)':>15} {'Candidate (B)':>15} {'Winner':>10}")
    print("-" * 52)
    for metric in ["f1", "auc"]:
        a, b = champion_metrics[metric], candidate_metrics[metric]
        winner = "A" if a >= b else "B"
        print(f"{metric.upper():<10} {a:>15.4f} {b:>15.4f} {winner:>10}")


if __name__ == "__main__":
    import glob
    archive_models = sorted(glob.glob(os.path.join(MODEL_DIR, "archive", "champion_model_*.pkl")))
    archive_encoders = sorted(glob.glob(os.path.join(MODEL_DIR, "archive", "encoders_*.pkl")))

    if not archive_models:
        print("No archived models found to compare against. Run retrain_gate.py first.")
    else:
        run(archive_models[-1], archive_encoders[-1])