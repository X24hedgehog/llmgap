#!/usr/bin/env python3
"""Pretty-print one datapoint from distractor_pairs_eedi.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "processed" / "distractor_pairs_eedi.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to distractor_pairs_eedi.csv",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="0-based row index to display",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample and print this many rows instead of using --index",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --sample-size",
    )
    return parser.parse_args()


def maybe_pretty_json(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        parsed = json.loads(text)
    except Exception:
        return value
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def print_row(row: dict, row_index: int, total_rows: int, csv_path: Path) -> None:
    print(f"CSV: {csv_path}")
    print(f"Row index: {row_index}")
    print(f"Total rows: {total_rows}")
    print("=" * 80)
    for key, value in row.items():
        print(f"{key}:")
        pretty_value = maybe_pretty_json(value)
        print(pretty_value)
        print("-" * 80)


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"CSV is empty: {args.csv}")

    if args.sample_size is not None:
        if args.sample_size <= 0:
            raise SystemExit("--sample-size must be a positive integer")

        sample_size = min(args.sample_size, len(df))
        sampled_df = df.sample(n=sample_size, random_state=args.seed)

        print(f"Random sample of {sample_size} rows from {args.csv}")
        print(f"Seed: {args.seed}")
        print("#" * 80)
        for position, (row_index, row_series) in enumerate(sampled_df.iterrows(), start=1):
            print(f"SAMPLED ROW {position}/{sample_size}")
            print_row(row_series.to_dict(), int(row_index), len(df), args.csv)
            if position != sample_size:
                print("\n")
        return

    row_index = args.index
    if row_index < 0 or row_index >= len(df):
        raise SystemExit(
            f"Index {row_index} is out of range for {args.csv} with {len(df)} rows"
        )

    row = df.iloc[row_index].to_dict()
    print_row(row, row_index, len(df), args.csv)


if __name__ == "__main__":
    main()