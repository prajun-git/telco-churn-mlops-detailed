import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt

from configs.config import MODEL_PATH, ENCODERS_PATH, TRAIN_PATH


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


def prepare_sample(encoders, n=200):
    """Use a sample of training data as the SHAP background/explanation set."""
    df = pd.read_csv(TRAIN_PATH).drop(columns=["customerID"]).copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    for col, le in encoders.items():
        df[col] = le.transform(df[col])
    X = df.drop(columns=["Churn"]).sample(n=n, random_state=42)
    return X


def run():
    print("Loading model and encoders...")
    model, encoders = load_artifacts()

    print("Preparing sample data...")
    X_sample = prepare_sample(encoders)

    print("Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values is a list [class_0, class_1] — use class 1 (churn)
    churn_shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values

    print("Generating global feature importance plot...")
    plt.figure()
    shap.summary_plot(churn_shap_values, X_sample, show=False)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "shap_summary.png")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"Saved global feature importance plot to: {output_path}")


if __name__ == "__main__":
    run()