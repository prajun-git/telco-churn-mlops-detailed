import sys
import os

# Allow importing from configs/ when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from configs.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_raw_data(path=RAW_DATA_PATH):
    """Load the raw Telco churn CSV."""
    return pd.read_csv(path)


def clean_data(df):
    """Fix TotalCharges dtype and impute the 11 tenure=0 blank rows."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isnull(), "TotalCharges"] = df.loc[
        df["TotalCharges"].isnull(), "MonthlyCharges"
    ]
    return df


def save_processed_data(df, path=PROCESSED_DATA_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def run():
    print("Loading raw data...")
    df = load_raw_data()
    print(f"Loaded shape: {df.shape}")

    print("Cleaning data...")
    df = clean_data(df)

    missing = df["TotalCharges"].isnull().sum()
    assert missing == 0, f"Unexpected missing values remain: {missing}"

    print("Saving processed data...")
    save_processed_data(df)
    print(f"Saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    run()