#!/usr/bin/env python3
"""Print EEDI reasoning-trace judgments marked as wrong for manual inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_FILES = [
    BASE_DIR / "output" / "eedi_reasoning_trace_eval" / "results" / "correct_answer_pairs_eedi_split_control_judged.csv",
    BASE_DIR / "output" / "eedi_reasoning_trace_eval" / "results" / "distractor_pairs_eedi_split_control_judged.csv",
]


def _load_wrong_rows(csv_path: Path, split: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "verdict" in df.columns:
        wrong_df = df[df["verdict"].astype(str).str.strip().str.lower() == "no"].copy()
    else:
        wrong_df = df[df["score"] == 0].copy()

    if split is not None and "split" in wrong_df.columns:
        wrong_df = wrong_df[wrong_df["split"] == split].copy()

    return wrong_df.reset_index(drop=True)


def _print_row(record: pd.Series, index: int, total: int) -> None:
    print(f"\n=== Wrong Example {index}/{total} ===")
    print(f"Task: {record.get('task', '')}")
    print(f"Question ID: {record.get('question_id', '')}")
    if "pair_index" in record and pd.notna(record["pair_index"]):
        print(f"Pair Index: {record['pair_index']}")
    if "split" in record and pd.notna(record["split"]):
        print(f"Split: {record['split']}")
    print(f"Verdict: {record.get('verdict', 'No')}")

    print("Question:")
    print(str(record.get("question", "")))

    print("\nGold label:")
    print(str(record.get("gold_answer", "")))

    print("\nReasoning trace:")
    print(str(record.get("reasoning_trace", "")))

    print("\nJudge output:")
    print(str(record.get("judge_output", "")))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print all EEDI reasoning-trace datapoints marked wrong for manual inspection.",
    )
    parser.add_argument(
        "--results-csv",
        nargs="+",
        default=[str(path) for path in DEFAULT_RESULT_FILES],
        help="One or more judged result CSVs. Defaults to both correct_answer and distractor outputs.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split filter, e.g. train / validation / test.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of wrong examples printed after filtering.",
    )
    args = parser.parse_args()

    all_wrong = []
    for raw_path in args.results_csv:
        csv_path = Path(raw_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing results CSV: {csv_path}")
        wrong_df = _load_wrong_rows(csv_path, args.split)
        if wrong_df.empty:
            print(f"No wrong rows found in {csv_path}")
            continue
        all_wrong.append(wrong_df)

    if not all_wrong:
        print("No wrong datapoints found.")
        return

    combined = pd.concat(all_wrong, ignore_index=True)
    if args.limit is not None:
        combined = combined.head(args.limit).copy()

    print(f"Printing {len(combined)} wrong datapoint(s).")
    for idx, (_, row) in enumerate(combined.iterrows(), start=1):
        _print_row(row, idx, len(combined))


if __name__ == "__main__":
    main()