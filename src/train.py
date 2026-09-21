import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from mlflow.models.signature import infer_signature


from configs.config import (
    TRAIN_PATH,
    VAL_PATH,
    MODEL_DIR,
    MODEL_PATH,
    ENCODERS_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_PARAMS,
)


def load_train_val():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    return train_df, val_df


def encode_features(train_df, val_df):
    """Drop customerID, encode target + categoricals. Fit encoders on train only."""
    train_df = train_df.drop(columns=["customerID"]).copy()
    val_df = val_df.drop(columns=["customerID"]).copy()

    train_df["Churn"] = train_df["Churn"].map({"Yes": 1, "No": 0})
    val_df["Churn"] = val_df["Churn"].map({"Yes": 1, "No": 0})

    categorical_cols = train_df.select_dtypes(include="object").columns.tolist()

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        val_df[col] = le.transform(val_df[col])
        encoders[col] = le

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_val = val_df.drop(columns=["Churn"])
    y_val = val_df["Churn"]

    return X_train, y_train, X_val, y_val, encoders


def train_model(X_train, y_train):
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    return model


def save_artifacts(model, encoders):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)


def run():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Loading train/val data...")
    train_df, val_df = load_train_val()

    print("Encoding features...")
    X_train, y_train, X_val, y_val, encoders = encode_features(train_df, val_df)

    print("Training model...")
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)

    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    with mlflow.start_run(run_name="pipeline_random_forest_balanced"):
        for param, value in MODEL_PARAMS.items():
            mlflow.log_param(param, value)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc)
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
    model,
    "model",
    signature=signature,
    input_example=X_train.iloc[:5],
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)
        run_id = mlflow.active_run().info.run_id

    print("Saving model + encoders to disk...")
    save_artifacts(model, encoders)

    print(f"Done. MLflow run ID: {run_id}")


if __name__ == "__main__":
    run()