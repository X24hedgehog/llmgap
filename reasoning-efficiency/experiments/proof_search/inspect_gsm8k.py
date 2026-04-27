"""
Inspect the GSM8K dataset (openai/gsm8k).

Usage:
    python inspect_gsm8k.py            # uses ROW_INDEX defined below
    python inspect_gsm8k.py --index 42 # override from CLI
"""

import argparse
import textwrap
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ── config ─────────────────────────────────────────────────────────────────────
ROW_INDEX = 0          # default row to inspect
SPLIT     = "train"    # "train" or "test"
TEXT_WIDTH = 100
# ───────────────────────────────────────────────────────────────────────────────


def wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(str(text), width=TEXT_WIDTH,
                         initial_indent=indent, subsequent_indent=indent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=ROW_INDEX,
                        help="Row index to inspect (default: %(default)s)")
    parser.add_argument("--split", default=SPLIT, choices=["train", "test"],
                        help="Dataset split to use (default: %(default)s)")
    args = parser.parse_args()

    # ── load ──────────────────────────────────────────────────────────────────
    print("Loading openai/gsm8k from local cache ...")
    parquet_dir = "/cluster/scratch/tunguyen1/hf_cache/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main"
    ds = load_dataset("parquet", data_files={
        "train": f"{parquet_dir}/train-00000-of-00001.parquet",
        "test":  f"{parquet_dir}/test-00000-of-00001.parquet",
    })

    # ── save to CSV ───────────────────────────────────────────────────────────
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    for split_name, split_ds in ds.items():
        out_path = out_dir / f"gsm8k_{split_name}.csv"
        if not out_path.exists():
            split_ds.to_pandas().to_csv(out_path, index=False)
            print(f"  Saved {len(split_ds):,} rows → {out_path}")
        else:
            print(f"  Already exists, skipping → {out_path}")

    # ── metadata ──────────────────────────────────────────────────────────────
    print("\n" + "═" * TEXT_WIDTH)
    print("  DATASET METADATA")
    print("═" * TEXT_WIDTH)
    for split_name, split_ds in ds.items():
        print(f"  split={split_name!r:10s}  rows={len(split_ds):>6,}  "
              f"columns={list(split_ds.column_names)}")

    # ── single row ────────────────────────────────────────────────────────────
    split_ds = ds[args.split]
    idx = args.index
    if idx < 0 or idx >= len(split_ds):
        raise IndexError(f"Index {idx} out of range for split '{args.split}' "
                         f"(size {len(split_ds)})")

    row = split_ds[idx]
    print(f"\n{'═' * TEXT_WIDTH}")
    print(f"  ROW {idx}  (split={args.split!r})")
    print("═" * TEXT_WIDTH)
    for col, val in row.items():
        print(f"\n{col}:")
        print(wrap(val))


if __name__ == "__main__":
    main()


