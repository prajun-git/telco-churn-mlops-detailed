import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split
from configs.config import (
    PROCESSED_DATA_PATH,
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
)


def load_processed_data(path=PROCESSED_DATA_PATH):
    return pd.read_csv(path)


def split_data(df):
    """Stratified 70/15/15 split on Churn."""
    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["Churn"],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=VAL_SIZE,
        stratify=temp_df["Churn"],
        random_state=RANDOM_STATE,
    )
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df):
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)


def run():
    print("Loading processed data...")
    df = load_processed_data()

    print("Splitting data (stratified 70/15/15)...")
    train_df, val_df, test_df = split_data(df)

    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        churn_pct = split_df["Churn"].value_counts(normalize=True)["Yes"] * 100
        print(f"{name}: {split_df.shape}, Churn%: {churn_pct:.2f}")

    print("Saving splits...")
    save_splits(train_df, val_df, test_df)
    print("Done.")


if __name__ == "__main__":
    run()