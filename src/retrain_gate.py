import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import pickle
from datetime import datetime

from configs.config import MODEL_PATH, ENCODERS_PATH, MODEL_DIR, VAL_PATH
from src import train as train_module
import pandas as pd
from sklearn.metrics import f1_score

# Minimum acceptable F1 drop before rejecting a new model (Step 5.4: e.g. accuracy drop < 1%)
MAX_ACCEPTABLE_F1_DROP = 0.02

def get_current_champion_f1():
    """Evaluate the currently deployed model on validation data."""
    if not os.path.exists(MODEL_PATH):
        return None  # no champion exists yet — first run

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    val_df = pd.read_csv(VAL_PATH).drop(columns=["customerID"]).copy()
    val_df["Churn"] = val_df["Churn"].map({"Yes": 1, "No": 0})
    for col, le in encoders.items():
        val_df[col] = le.transform(val_df[col])

    X_val = val_df.drop(columns=["Churn"])
    y_val = val_df["Churn"]

    return f1_score(y_val, model.predict(X_val))


def backup_current_champion():
    """Keep the previous model for rollback (Step 8.5)."""
    if not os.path.exists(MODEL_PATH):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(MODEL_DIR, "archive")
    os.makedirs(backup_dir, exist_ok=True)

    backup_model_path = os.path.join(backup_dir, f"champion_model_{timestamp}.pkl")
    backup_encoders_path = os.path.join(backup_dir, f"encoders_{timestamp}.pkl")

    shutil.copy(MODEL_PATH, backup_model_path)
    shutil.copy(ENCODERS_PATH, backup_encoders_path)

    print(f"Backed up current champion to: {backup_model_path}")
    return backup_model_path


def run():
    print("Checking current champion performance (if any)...")
    old_f1 = get_current_champion_f1()

    if old_f1 is not None:
        print(f"Current champion F1: {old_f1:.4f}")
    else:
        print("No existing champion — this will be the first model.")

    # Back up old model BEFORE training overwrites it
    backup_current_champion()

    print("\nTraining candidate model...")
    train_module.run()  # this trains and overwrites champion_model.pkl

    with open(MODEL_PATH, "rb") as f:
        new_model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        new_encoders = pickle.load(f)

    val_df = pd.read_csv(VAL_PATH).drop(columns=["customerID"]).copy()
    val_df["Churn"] = val_df["Churn"].map({"Yes": 1, "No": 0})
    for col, le in new_encoders.items():
        val_df[col] = le.transform(val_df[col])
    X_val = val_df.drop(columns=["Churn"])
    y_val = val_df["Churn"]
    new_f1 = f1_score(y_val, new_model.predict(X_val))

    print(f"\nCandidate model F1: {new_f1:.4f}")

    if old_f1 is not None:
        drop = old_f1 - new_f1
        if drop > MAX_ACCEPTABLE_F1_DROP:
            print(f"\nREJECTED: F1 dropped by {drop:.4f} (max acceptable: {MAX_ACCEPTABLE_F1_DROP})")
            print("Rolling back to previous champion...")
            # Restore from the backup we just made
            backups = sorted(os.listdir(os.path.join(MODEL_DIR, "archive")))
            latest_model_backup = [b for b in backups if b.startswith("champion_model_")][-1]
            latest_encoders_backup = [b for b in backups if b.startswith("encoders_")][-1]
            shutil.copy(os.path.join(MODEL_DIR, "archive", latest_model_backup), MODEL_PATH)
            shutil.copy(os.path.join(MODEL_DIR, "archive", latest_encoders_backup), ENCODERS_PATH)
            print("Rollback complete. Previous champion restored.")
        else:
            print(f"\nACCEPTED: New model registered as champion (F1 change: {-drop:+.4f})")
    else:
        print("\nACCEPTED: First model registered as champion.")


if __name__ == "__main__":
    run()