#!/usr/bin/env python3
"""Inspect scores from inference-before results in out/interim/.

For correct_answer: reports accuracy (mean of binary score).
For distractor: reports accuracy from the existing score column,
  and notes that TP/FP/FN/Jaccard are not yet available (pending
  the run_inference.py fix to use set-level scoring).

Usage:
    python inspect_scores.py
    python inspect_scores.py --interim-dir out/interim
    python inspect_scores.py --task distractor
    python inspect_scores.py --model Qwen2.5-0.5B-Instruct
"""
import argparse
import os
import re

import pandas as pd


def parse_filename(fname: str):
    """Extract (model, task, mode) from filename like 'Qwen2.5-0.5B-Instruct_correct_answer_before.csv'."""
    m = re.match(r"^(.+)_(correct_answer|distractor)_(before|after)\.csv$", fname)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None


def summarize(df: pd.DataFrame, model: str, task: str, mode: str):
    n = len(df)
    result = {
        "model": model,
        "task": task,
        "mode": mode,
        "n_rows": n,
    }
    # Set-level metrics (tp, fp, fn, jaccard)
    if "tp" in df.columns and "fp" in df.columns and "fn" in df.columns:
        result["format"] = "set"
        result["total_tp"] = int(df["tp"].sum())
        result["total_fp"] = int(df["fp"].sum())
        result["total_fn"] = int(df["fn"].sum())
        tp = result["total_tp"]
        fp = result["total_fp"]
        fn = result["total_fn"]
        result["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
        result["recall"] = tp / (tp + fn) if (tp + fn) else 0.0
        result["mean_jaccard"] = df["jaccard"].mean() if "jaccard" in df.columns else 0.0
    # Simple binary score
    elif "score" in df.columns:
        result["format"] = "binary"
        result["accuracy"] = df["score"].mean()
        result["correct"] = int(df["score"].sum())
        result["incorrect"] = int(n - df["score"].sum())
    else:
        result["format"] = "unknown"
    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect inference scores from interim CSVs.")
    parser.add_argument("--interim-dir", default="out/interim")
    parser.add_argument("--task", default=None, choices=["correct_answer", "distractor"],
                        help="Filter to a specific task")
    parser.add_argument("--model", default=None,
                        help="Filter to a specific model (substring match)")
    parser.add_argument("--mode", default=None, choices=["before", "after"],
                        help="Filter to before or after finetuning")
    args = parser.parse_args()

    csvs = sorted(f for f in os.listdir(args.interim_dir) if f.endswith(".csv"))
    if not csvs:
        print(f"No CSV files found in {args.interim_dir}")
        return

    results = []
    for fname in csvs:
        model, task, mode = parse_filename(fname)
        if model is None:
            continue
        if args.task and task != args.task:
            continue
        if args.model and args.model.lower() not in model.lower():
            continue
        if args.mode and mode != args.mode:
            continue

        df = pd.read_csv(os.path.join(args.interim_dir, fname))
        results.append(summarize(df, model, task, mode))

    if not results:
        print("No matching results found.")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["task", "mode", "model"]).reset_index(drop=True)

    # Print grouped by task
    for task in results_df["task"].unique():
        task_df = results_df[results_df["task"] == task]
        print(f"\n{'='*80}")
        print(f"  TASK: {task}")
        print(f"{'='*80}")

        for mode in sorted(task_df["mode"].unique()):
            mode_df = task_df[task_df["mode"] == mode].copy()
            fmt = mode_df.iloc[0]["format"] if len(mode_df) > 0 else "binary"

            print(f"\n  Mode: {mode}")
            print(f"  {'─'*86}")

            if fmt == "set":
                print(f"  {'Model':<40} {'Rows':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'Jaccard':>8}")
                print(f"  {'─'*86}")
                for _, row in mode_df.iterrows():
                    print(f"  {row['model']:<40} {row['n_rows']:>6} {row['total_tp']:>6} {row['total_fp']:>6} "
                          f"{row['total_fn']:>6} {row['precision']:>8.4f} {row['recall']:>8.4f} {row['mean_jaccard']:>8.4f}")
                print(f"  {'─'*86}")
                mean_prec = mode_df["precision"].mean()
                mean_rec = mode_df["recall"].mean()
                mean_jac = mode_df["mean_jaccard"].mean()
                print(f"  {'MEAN across models':<40} {'':>6} {'':>6} {'':>6} "
                      f"{'':>6} {mean_prec:>8.4f} {mean_rec:>8.4f} {mean_jac:>8.4f}")
            else:
                print(f"  {'Model':<40} {'Rows':>6} {'Correct':>8} {'Incorrect':>10} {'Accuracy':>10}")
                print(f"  {'─'*86}")
                for _, row in mode_df.iterrows():
                    print(f"  {row['model']:<40} {row['n_rows']:>6} {row.get('correct', 'N/A'):>8} "
                          f"{row.get('incorrect', 'N/A'):>10} {row.get('accuracy', 0):>10.4f}")
                print(f"  {'─'*86}")
                total_rows = mode_df["n_rows"].sum()
                total_correct = mode_df.get("correct", pd.Series([0])).sum()
                mean_acc = mode_df["accuracy"].mean() if "accuracy" in mode_df.columns else 0
                print(f"  {'MEAN across models':<40} {total_rows:>6} {int(total_correct):>8} "
                      f"{int(total_rows - total_correct):>10} {mean_acc:>10.4f}")

    print()


if __name__ == "__main__":
    main()
