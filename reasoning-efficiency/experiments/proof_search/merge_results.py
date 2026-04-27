#!/usr/bin/env python3
"""Merge interim inference results into model columns on the main task CSVs.

For each (model, task) that has both a 'before' and 'after' interim CSV in
out/interim/, this script adds (or overwrites) a column named <model_short> in
the corresponding main task CSV.

Column format (JSON string):
    {
        "before": [prediction_before_ft, score_before],
        "after":  [prediction_after_ft,  score_after]
    }

where score_before / score_after are 0 or 1.
Rows outside the test split get null for that column (train rows were not inferred).

Run after all run_inference.py jobs complete:
    python merge_results.py
"""
import argparse
import glob
import json
import os

import pandas as pd
from collections import defaultdict
from tqdm import tqdm

TASK_CSV = {
    "correct_answer": "out/correct_answer_pairs.csv",
    "next_subquestion": "out/next_subquestion_pairs.csv",
}

# Standard columns that are never model columns
_BASE_COLS = {
    "example_idx", "problem_type", "complexity", "overlap_type", "problem",
    "question", "groundquery", "reasoning_trace", "answer", "depth", "width",
    "subquestions", "subquestions_by_node_id", "subquestion_tree", "trees",
    "pair_index", "solved_node_ids", "solved_steps", "target_node_id",
    "next_subquestion", "prompt",
    "target_question", "target_problem", "target_answer",  # correct_answer task
    "split",
}


def _discover_interim(interim_dir: str) -> dict:
    """Return {(model_short, task): {mode: DataFrame}} from files in interim_dir."""
    groups = defaultdict(dict)
    for fpath in glob.glob(os.path.join(interim_dir, "*.csv")):
        fname = os.path.basename(fpath).replace(".csv", "")
        for task in ("correct_answer", "next_subquestion"):
            for mode in ("before", "after"):
                suffix = f"_{task}_{mode}"
                if fname.endswith(suffix):
                    model_short = fname[: -len(suffix)]
                    groups[(model_short, task)][mode] = pd.read_csv(fpath)
    return groups


def merge(interim_dir: str) -> None:
    groups = _discover_interim(interim_dir)
    if not groups:
        print(f"No interim CSVs found in {interim_dir}")
        return

    for task, csv_path in TASK_CSV.items():
        if not os.path.exists(csv_path):
            print(f"[SKIP] Main CSV not found: {csv_path}")
            continue

        main_df = pd.read_csv(csv_path)
        updated_any = False

        for (model_short, t), mode_dfs in groups.items():
            if t != task:
                continue
            if "before" not in mode_dfs or "after" not in mode_dfs:
                print(
                    f"[SKIP] {model_short}/{task}: "
                    f"only {list(mode_dfs.keys())} found, need both before+after"
                )
                continue

            # Index interim DataFrames by (example_idx, pair_index) for fast lookup
            before_idx = mode_dfs["before"].set_index(["example_idx", "pair_index"])
            after_idx = mode_dfs["after"].set_index(["example_idx", "pair_index"])

            col_values = []
            for _, row in tqdm(main_df.iterrows(), total=len(main_df),
                               desc=f"Merging {model_short}/{task}"):
                key = (row["example_idx"], row["pair_index"])
                if key in before_idx.index and key in after_idx.index:
                    b = before_idx.loc[key]
                    a = after_idx.loc[key]
                    val = json.dumps({
                        "before": [str(b["prediction"]), int(b["score"])],
                        "after":  [str(a["prediction"]), int(a["score"])],
                    })
                else:
                    val = None  # train row or missing
                col_values.append(val)

            main_df[model_short] = col_values
            n_filled = sum(v is not None for v in col_values)
            print(f"  [OK] '{model_short}' → {csv_path}  ({n_filled} rows filled)")
            updated_any = True

        if updated_any:
            main_df.to_csv(csv_path, index=False)
            print(f"  Saved {csv_path}\n")
        else:
            print(f"  No new columns for {csv_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge interim inference CSVs into model columns in the main task CSVs."
    )
    parser.add_argument("--interim-dir", default="out/interim",
                        help="Directory containing interim result CSVs (default: out/interim)")
    args = parser.parse_args()
    merge(args.interim_dir)


if __name__ == "__main__":
    main()
