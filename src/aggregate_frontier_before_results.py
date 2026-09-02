#!/usr/bin/env python3
"""Aggregate run_inference_frontier.py --mode before summary JSONs into a
per-model, per-(setting, task) accuracy table, plus the accuracy gap across
tasks for each model.

Usage:
    python aggregate_frontier_before_results.py --results-dir out/frontier_eval
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def load_summaries(results_dir: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(results_dir.glob("*_before_*_summary.json")):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("mode") != "before":
            continue

        # suffix was written as "_<setting>" by run_frontier_before_all.sh
        stem = summary_path.stem  # "<model>_<task>_before_<setting>_summary"
        setting = stem.rsplit("_summary", 1)[0].rsplit("_", 1)[-1]

        rows.append({
            "model": summary["model_name"],
            "setting": setting,
            "task": summary["task"],
            "n": summary["n"],
            "accuracy": summary["accuracy"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="out/frontier_eval")
    args = parser.parse_args()

    df = load_summaries(Path(args.results_dir))
    if df.empty:
        print(f"No before-mode summary files found in {args.results_dir}")
        return

    pivot = df.pivot_table(
        index="model", columns=["setting", "task"], values="accuracy"
    )
    print("=" * 100)
    print("Accuracy by model x (setting, task) — before fine-tuning")
    print("=" * 100)
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))

    gap = pivot.max(axis=1) - pivot.min(axis=1)
    print("\nGap between tasks per model (max accuracy - min accuracy):")
    print(gap.sort_values(ascending=False).to_string(float_format=lambda v: f"{v:.3f}"))

    out_csv = Path(args.results_dir) / "before_only_task_gap.csv"
    pivot.to_csv(out_csv)
    print(f"\nSaved full table → {out_csv}")


if __name__ == "__main__":
    main()
