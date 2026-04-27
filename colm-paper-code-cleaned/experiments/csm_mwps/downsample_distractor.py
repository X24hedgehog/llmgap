"""Downsample both correct_answer and distractor CSVs to target sizes.

Distractor CSV is stratified by (split, number_of_distractors).
Correct-answer CSV is simple random sampling per split.

Usage:
  python downsample_distractor.py --ca-csv out/correct_answer_distractor_pairs.csv \
                                  --dist-csv out/distractor_pairs.csv \
                                  --n-train 10000 --n-test 2000 --seed 14
"""

import argparse
import json

import pandas as pd


def _downsample_simple(df, n_train, n_test, seed):
    """Random sample per split."""
    parts = []
    for split_val, target in [("train", n_train), ("test", n_test)]:
        split_df = df[df["split"] == split_val]
        n = min(target, len(split_df))
        parts.append(split_df.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def _downsample_stratified(df, n_train, n_test, seed):
    """Stratified sample by (split, n_distractors)."""
    df["_n_dist"] = df["distractor_answers"].apply(lambda x: len(json.loads(x)))
    parts = []
    for split_val, target in [("train", n_train), ("test", n_test)]:
        split_df = df[df["split"] == split_val]
        group_counts = split_df.groupby("_n_dist").size()
        fracs = group_counts / group_counts.sum()
        allocs = (fracs * target).apply(int)
        remainder = target - allocs.sum()
        if remainder > 0:
            for idx in fracs.sort_values(ascending=False).index:
                if remainder <= 0:
                    break
                allocs[idx] += 1
                remainder -= 1
        for n_dist, n_sample in allocs.items():
            bucket = split_df[split_df["_n_dist"] == n_dist]
            parts.append(bucket.sample(n=min(n_sample, len(bucket)),
                                       random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    out.drop(columns=["_n_dist"], inplace=True)
    return out


def _stats(df):
    tr = (df["split"] == "train").sum()
    te = (df["split"] == "test").sum()
    return f"{len(df)} ({tr} train / {te} test)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ca-csv", required=True)
    parser.add_argument("--dist-csv", required=True)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()

    ca_df = pd.read_csv(args.ca_csv)
    dist_df = pd.read_csv(args.dist_csv)
    print(f"Before:  ca={_stats(ca_df)},  dist={_stats(dist_df)}")

    ca_df = _downsample_simple(ca_df, args.n_train, args.n_test, args.seed)
    dist_df = _downsample_stratified(dist_df, args.n_train, args.n_test, args.seed)

    ca_df.to_csv(args.ca_csv, index=False)
    dist_df.to_csv(args.dist_csv, index=False)
    print(f"After:   ca={_stats(ca_df)},  dist={_stats(dist_df)}")


if __name__ == "__main__":
    main()
