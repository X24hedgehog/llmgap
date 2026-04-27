#!/usr/bin/env python3
"""
Inspect rows from an interim result CSV, with prompt and gold joined in.

Usage:
    python inspect_interim.py out/interim/Qwen2.5-0.5B-Instruct_correct_answer_before.csv
    python inspect_interim.py out/interim/Qwen2.5-0.5B-Instruct_next_subquestion_before.csv
    python inspect_interim.py out/interim/Qwen2.5-0.5B-Instruct_correct_answer_before.csv --rows 20
    python inspect_interim.py out/interim/Qwen2.5-0.5B-Instruct_correct_answer_before.csv --wrong-only
    python inspect_interim.py out/interim/Qwen2.5-0.5B-Instruct_correct_answer_before.csv --idx 3 7 12
"""

import argparse
import os
import pandas as pd

TASK_CSVS = {
    "correct_answer":  "out/correct_answer_pairs.csv",
    "next_subquestion": "out/next_subquestion_pairs.csv",
}
GOLD_COL = {
    "correct_answer":  "target_answer",
    "next_subquestion": "next_subquestion",
}

def infer_task(filename: str) -> str:
    if "correct_answer" in filename:
        return "correct_answer"
    if "next_subquestion" in filename:
        return "next_subquestion"
    raise ValueError(f"Cannot infer task from filename: {filename}")

def print_row(i: int, row: pd.Series, gold_col: str, width: int = 80):
    sep = "─" * width
    print(sep)
    print(f"  Row {i}  │  example_idx={row['example_idx']}  pair_index={row['pair_index']}  │  score={row['score']}")
    print(sep)
    print(f"PROMPT:\n{row['prompt']}")
    print()
    print(f"GOLD    : {row[gold_col]}")
    print(f"PREDICT : {row['prediction']}")
    print(f"SCORE   : {row['score']}")
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to interim CSV (out/interim/*.csv)")
    parser.add_argument("--rows", type=int, default=10,
                        help="Number of rows to show (default: 10)")
    parser.add_argument("--wrong-only", action="store_true",
                        help="Show only rows where score=0")
    parser.add_argument("--correct-only", action="store_true",
                        help="Show only rows where score=1")
    parser.add_argument("--idx", type=int, nargs="+",
                        help="Show specific row indices (0-based, overrides --rows)")
    args = parser.parse_args()

    # ── load interim results ──────────────────────────────────────────────────
    interim = pd.read_csv(args.csv)
    print(f"Loaded {len(interim)} rows from {args.csv}")
    print(f"Overall accuracy: {interim['score'].mean():.4f}  ({interim['score'].sum()}/{len(interim)})\n")

    # ── infer task and load main csv for prompt + gold ────────────────────────
    task = infer_task(os.path.basename(args.csv))
    gold_col = GOLD_COL[task]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_csv = os.path.join(script_dir, TASK_CSVS[task])
    main_df = pd.read_csv(main_csv, usecols=["example_idx", "pair_index", "prompt", gold_col])
    merged = interim.merge(main_df, on=["example_idx", "pair_index"], how="left")

    # ── filter ────────────────────────────────────────────────────────────────
    if args.idx:
        subset = merged.iloc[args.idx]
        for i, (_, row) in zip(args.idx, subset.iterrows()):
            print_row(i, row, gold_col)
        return

    if args.wrong_only:
        subset = merged[merged["score"] == 0]
        label = "wrong"
    elif args.correct_only:
        subset = merged[merged["score"] == 1]
        label = "correct"
    else:
        subset = merged
        label = "all"

    subset = subset.head(args.rows)
    print(f"Showing {len(subset)} {label} rows:\n")
    for i, (_, row) in enumerate(subset.iterrows()):
        print_row(i, row, gold_col)

if __name__ == "__main__":
    main()
