import os

# Project root (one level up from configs/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "telco_churn_cleaned.csv")

TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "data", "processed", "val.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")

# Model artifact paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "champion_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")

# MLflow
MLFLOW_TRACKING_URI = f"file:{os.path.join(BASE_DIR, 'mlruns')}"
MLFLOW_EXPERIMENT_NAME = "telco-churn-prediction"

# Split config
RANDOM_STATE = 42
TEST_SIZE = 0.30   # first split: 70% train / 30% temp
VAL_SIZE = 0.50    # second split: temp → 15% val / 15% test

# Model config (champion: class-weighted Random Forest)
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}