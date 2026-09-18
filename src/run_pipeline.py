import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_prep, split_data, train, evaluate


def run_pipeline():
    print("=" * 50)
    print("STEP 1/4: Data Preparation")
    print("=" * 50)
    data_prep.run()

    print("\n" + "=" * 50)
    print("STEP 2/4: Train/Val/Test Split")
    print("=" * 50)
    split_data.run()

    print("\n" + "=" * 50)
    print("STEP 3/4: Model Training")
    print("=" * 50)
    train.run()

    print("\n" + "=" * 50)
    print("STEP 4/4: Evaluation on Test Set")
    print("=" * 50)
    evaluate.run(split="test")

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()