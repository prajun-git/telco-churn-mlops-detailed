import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix

from configs.config import MODEL_PATH, ENCODERS_PATH, VAL_PATH, TEST_PATH


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


def prepare_data(df, encoders):
    df = df.drop(columns=["customerID"]).copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print("F1-score:", f1_score(y, y_pred))
    print("AUC-ROC:", roc_auc_score(y, y_proba))
    print()
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


def run(split="test"):
    path_map = {"val": VAL_PATH, "test": TEST_PATH}
    if split not in path_map:
        raise ValueError(f"split must be 'val' or 'test', got '{split}'")

    print(f"Evaluating on: {split}")
    model, encoders = load_artifacts()

    df = pd.read_csv(path_map[split])
    X, y = prepare_data(df, encoders)

    evaluate(model, X, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    run(split=args.split)