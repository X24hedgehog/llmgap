#!/usr/bin/env python3
"""Compute per-datapoint performance gain from model columns in the main task CSVs.

After merge_results.py has written model columns, this script:
  - Filters to test-split rows
  - For each model column, reads before/after scores from the stored JSON
  - Computes acc_before, acc_after, and gain = mean(score_after - score_before)

Run:
    python compute_gain.py
"""
import argparse
import json
import os

import pandas as pd

TASK_CSV = {
    "correct_answer": "out/correct_answer_pairs.csv",
    "next_subquestion": "out/next_subquestion_pairs.csv",
}

# Standard columns — everything else is treated as a model result column
_BASE_COLS = {
    "example_idx", "problem_type", "complexity", "overlap_type", "problem",
    "question", "groundquery", "reasoning_trace", "answer", "depth", "width",
    "subquestions", "subquestions_by_node_id", "subquestion_tree", "trees",
    "pair_index", "solved_node_ids", "solved_steps", "target_node_id",
    "next_subquestion", "prompt",
    "target_question", "target_problem", "target_answer",
    "split",
}


def compute_stats() -> pd.DataFrame:
    rows = []
    for task, csv_path in TASK_CSV.items():
        if not os.path.exists(csv_path):
            print(f"[SKIP] {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        if "split" not in df.columns:
            print(f"[WARN] No 'split' column in {csv_path} — run split_data.py first")
            continue

        test_df = df[df["split"] == "test"]
        model_cols = [c for c in df.columns if c not in _BASE_COLS]

        if not model_cols:
            print(f"[INFO] No model columns found in {csv_path} — run merge_results.py first")
            continue

        print(f"[{task}] {len(test_df)} test rows, {len(model_cols)} model(s): {model_cols}")

        for col in model_cols:
            before_scores, after_scores = [], []
            for val in test_df[col].dropna():
                try:
                    d = json.loads(val)
                    before_scores.append(int(d["before"][1]))
                    after_scores.append(int(d["after"][1]))
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue

            if not before_scores:
                print(f"  [WARN] No valid entries for column '{col}'")
                continue

            acc_before = sum(before_scores) / len(before_scores)
            acc_after = sum(after_scores) / len(after_scores)
            gain = sum(a - b for a, b in zip(after_scores, before_scores)) / len(before_scores)

            rows.append(dict(
                model=col,
                task=task,
                acc_before=round(acc_before, 4),
                acc_after=round(acc_after, 4),
                gain=round(gain, 4),
                n_test=len(before_scores),
            ))

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["model", "task", "acc_before", "acc_after", "gain", "n_test"]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute performance gain from model columns in the main task CSVs."
    )
    parser.add_argument("--out-csv", default="out/results/gain_summary.csv",
                        help="Where to write the summary table")
    args = parser.parse_args()

    result_df = compute_stats()

    if result_df.empty:
        print("No results to report yet.")
        return

    print()
    for task in TASK_CSV:
        sub = result_df[result_df["task"] == task]
        if sub.empty:
            continue
        print(f"=== Task: {task} ===")
        print(sub[["model", "acc_before", "acc_after", "gain", "n_test"]].to_string(index=False))
        print()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    result_df.to_csv(args.out_csv, index=False)
    print(f"Saved gain summary → {args.out_csv}")


if __name__ == "__main__":
    main()
