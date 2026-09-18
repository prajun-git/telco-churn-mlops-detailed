import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd

from configs.config import TRAIN_PATH

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "predictions.jsonl")

# Numeric columns worth monitoring for drift
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Alert if a feature's mean shifts by more than this fraction (Step 7.4: e.g. 20%)
DRIFT_THRESHOLD = 0.20


def load_training_stats():
    df = pd.read_csv(TRAIN_PATH)
    return df[NUMERIC_COLS].mean()


def load_logged_inputs():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"No prediction logs found at {LOG_PATH}")

    records = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            entry = json.loads(line)
            records.append(entry["input"])

    return pd.DataFrame(records)


def check_drift(train_means, logged_df):
    print(f"{'Feature':<20} {'Train Mean':>12} {'Live Mean':>12} {'% Shift':>10} {'Status':>10}")
    print("-" * 68)

    any_drift = False
    for col in NUMERIC_COLS:
        train_mean = train_means[col]
        live_mean = logged_df[col].mean()
        pct_shift = abs(live_mean - train_mean) / train_mean

        status = "DRIFT" if pct_shift > DRIFT_THRESHOLD else "OK"
        if status == "DRIFT":
            any_drift = True

        print(f"{col:<20} {train_mean:>12.2f} {live_mean:>12.2f} {pct_shift:>9.1%} {status:>10}")

    print()
    if any_drift:
        print(f"ALERT: Drift threshold ({DRIFT_THRESHOLD:.0%}) exceeded on one or more features.")
    else:
        print("No significant drift detected.")


def run():
    print("Loading training baseline...")
    train_means = load_training_stats()

    print("Loading logged predictions...")
    logged_df = load_logged_inputs()
    print(f"{len(logged_df)} logged predictions found.\n")

    check_drift(train_means, logged_df)


if __name__ == "__main__":
    run()