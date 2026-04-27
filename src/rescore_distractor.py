#!/usr/bin/env python3
"""Re-score existing distractor predictions using the updated 4-shot judge prompt.

Reads each *_distractor_{before,after}.csv in the distractor interim folder,
re-runs the judge on stored predictions, and overwrites the score column.

Usage:
    python src/rescore_distractor.py [--batch-size 8] [--dry-run]
"""
import argparse
import csv
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the updated score_distractor and helpers from run_inference
sys.path.insert(0, os.path.dirname(__file__))
from run_inference import score_distractor, TASK_TARGET_COL

DATA_CSV = os.path.join(
    os.path.dirname(__file__), "..",
    "colm-paper-code-cleaned/experiments/csm_mwps/out/distractor_pairs.csv",
)
INTERIM_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "out/distractor/results/interim",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be rescored without running judge")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit rows per CSV (for testing, e.g. --max-rows 100)")
    parser.add_argument("--csv-filter", default=None,
                        help="Only rescore CSVs matching this substring (e.g. 'Qwen2.5-1.5B')")
    args = parser.parse_args()

    # Read source CSV to get gold answers (test split only)
    with open(DATA_CSV, newline="") as f:
        source_rows = [r for r in csv.DictReader(f) if r.get("split") == "test"]
    target_col = TASK_TARGET_COL["distractor"]

    # Find all distractor CSVs
    csv_files = sorted(
        f for f in os.listdir(INTERIM_DIR)
        if "_distractor_" in f and f.endswith(".csv")
        and (args.csv_filter is None or args.csv_filter in f)
    )

    if not csv_files:
        print("No distractor CSVs found in", INTERIM_DIR)
        return

    print(f"Found {len(csv_files)} distractor CSV(s) to rescore:")
    for f in csv_files:
        print(f"  {f}")

    if args.dry_run:
        print("\n--dry-run: exiting without rescoring.")
        return

    for csv_name in csv_files:
        csv_path = os.path.join(INTERIM_DIR, csv_name)

        # Read result CSV
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            result_rows = list(reader)

        n = min(len(result_rows), len(source_rows))
        if args.max_rows is not None:
            n = min(n, args.max_rows)
        predictions = [result_rows[i]["prediction"] for i in range(n)]
        golds = [source_rows[i][target_col] for i in range(n)]

        old_scores = [int(result_rows[i]["score"]) for i in range(n)]
        old_acc = sum(old_scores) / len(old_scores)

        print(f"\n{'─' * 70}")
        print(f"Rescoring: {csv_name}  ({n} rows, old accuracy: {old_acc:.1%})")

        new_scores = score_distractor(predictions, golds, args.batch_size)

        new_acc = sum(new_scores) / len(new_scores)
        changed = sum(1 for a, b in zip(old_scores, new_scores) if a != b)
        gained = sum(1 for a, b in zip(old_scores, new_scores) if a == 0 and b == 1)
        lost = sum(1 for a, b in zip(old_scores, new_scores) if a == 1 and b == 0)

        print(f"  New accuracy: {new_acc:.1%}  (Δ {new_acc - old_acc:+.1%})")
        print(f"  Changed: {changed}  (gained: +{gained}, lost: -{lost})")

        if args.max_rows is not None:
            print(f"  (test mode: --max-rows {args.max_rows}, NOT writing to file)")
            continue

        # Update scores in result rows
        for i in range(n):
            result_rows[i]["score"] = new_scores[i]

        # Write back
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(result_rows)

        print(f"  Saved → {csv_path}")

    print(f"\n{'═' * 70}")
    print("Done rescoring all distractor CSVs.")


if __name__ == "__main__":
    main()
