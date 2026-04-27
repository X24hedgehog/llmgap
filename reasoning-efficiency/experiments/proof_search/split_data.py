#!/usr/bin/env python3
"""Add a 'split' column (train / test) in-place to both main task CSVs.

Splits are determined by example_idx (not individual rows) to prevent leakage.
Run once before any fine-tuning or inference.

Usage:
    python split_data.py --seed 42 --test-ratio 0.2
"""
import argparse

import numpy as np
import pandas as pd

TASK_CSVS = [
    "out/correct_answer_pairs.csv",
    "out/next_subquestion_pairs.csv",
]


def add_split_column(seed: int, test_ratio: float) -> None:
    # Determine test/train split from example_ids in first CSV (same across both)
    df0 = pd.read_csv(TASK_CSVS[0])
    example_ids = sorted(df0["example_idx"].unique())
    n = len(example_ids)
    n_test = max(1, int(n * test_ratio))

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(example_ids)
    test_ids = set(shuffled[:n_test].tolist())

    print(f"Total base examples: {n}  → train: {n - len(test_ids)}, test: {len(test_ids)}")

    for csv_path in TASK_CSVS:
        df = pd.read_csv(csv_path)
        df["split"] = df["example_idx"].apply(
            lambda x: "test" if x in test_ids else "train"
        )
        df.to_csv(csv_path, index=False)
        n_train_rows = (df["split"] == "train").sum()
        n_test_rows = (df["split"] == "test").sum()
        print(f"  {csv_path}: {n_train_rows} train rows, {n_test_rows} test rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of base examples to hold out for test (default: 0.2)")
    args = parser.parse_args()
    add_split_column(args.seed, args.test_ratio)
