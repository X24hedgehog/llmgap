#!/usr/bin/env python3
"""Build a curated non-geometry misconception list from the raw EEDI mapping.

This intentionally avoids the previous OpenAI-based geometry labels and applies
local deterministic rules instead. The goal is to keep non-geometry topics such
as arithmetic, algebra, graphs, statistics, and probability, while excluding
clearly geometry- and shape-focused misconceptions.

Outputs:
  - data/processed/misconceptions.csv
  - data/processed/allowed_misconceptions.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = ROOT / "data" / "raw" / "misconception_mapping.csv"
DEFAULT_OUTPUT_CSV = ROOT / "data" / "processed" / "misconceptions.csv"
DEFAULT_COMPAT_OUTPUT_CSV = ROOT / "data" / "processed" / "allowed_misconceptions.csv"

KEEP_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"pie chart",
        r"bar chart",
        r"histogram",
        r"scatter graph",
        r"line of best fit",
        r"cumulative frequency",
        r"frequency density",
        r"speed-time graph",
        r"distance-time graph",
        r"inverse proportion",
        r"linear equation",
        r"quadratic graph",
        r"turning point",
        r"sample size",
        r"stratified sample",
        r"sample space",
        r"probability",
        r"ratio",
        r"fraction",
        r"decimal",
        r"percentage",
        r"algebra",
        r"equation",
        r"expression",
        r"function machine",
        r"function notation",
        r"sequence",
        r"graph",
        r"gradient",
        r"coordinates of the turning point",
        r"x and y coordinates",
        r"area model",
        r"bar model",
    ]
]

DROP_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bangle(?:s)?\b",
        r"triangle",
        r"rectangle",
        r"rectilinear",
        r"square shape",
        r"quadrilateral",
        r"polygon",
        r"pentagon",
        r"hexagon",
        r"octagon",
        r"decagon",
        r"regular\s+\d+\s+sided",
        r"\bcircle(?:s)?\b",
        r"circumference",
        r"\bradius\b",
        r"diameter",
        r"arc",
        r"chord",
        r"tangent",
        r"sector",
        r"segment",
        r"perimeter",
        r"surface area",
        r"\bvolume\b",
        r"prism",
        r"cuboid",
        r"cylinder",
        r"cone",
        r"sphere",
        r"net\b",
        r"symmetry",
        r"parallel lines?",
        r"perpendicular",
        r"vertices|vertex",
        r"edges?|faces?",
        r"interior angle",
        r"exterior angle",
        r"full turn",
        r"straight line",
        r"acute angle",
        r"obtuse angle",
        r"reflex angle",
        r"diagonal",
        r"line segment",
        r"distance between\s+\d+\s+points",
        r"enlarg",
        r"reflection",
        r"rotation",
        r"translation",
        r"transform",
        r"bearing",
        r"loci|locus",
        r"construction",
        r"trigon",
        r"pythag",
        r"vector",
        r"shape",
    ]
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--compat-output-csv",
        type=Path,
        default=DEFAULT_COMPAT_OUTPUT_CSV,
        help="Optional compatibility output matching the previous filename",
    )
    return parser.parse_args()


def should_keep_misconception(name: str) -> bool:
    text = str(name).strip()
    if not text:
        return False

    if any(pattern.search(text) for pattern in KEEP_PATTERNS):
        return True

    if any(pattern.search(text) for pattern in DROP_PATTERNS):
        return False

    return True


def main() -> None:
    args = parse_args()
    mapping_df = pd.read_csv(args.input_csv)
    keep_mask = mapping_df["MisconceptionName"].map(should_keep_misconception)
    kept_df = mapping_df.loc[keep_mask, ["MisconceptionId", "MisconceptionName"]].copy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    kept_df.to_csv(args.output_csv, index=False)
    if args.compat_output_csv:
        args.compat_output_csv.parent.mkdir(parents=True, exist_ok=True)
        kept_df.to_csv(args.compat_output_csv, index=False)

    print(f"Total misconceptions: {len(mapping_df):,}")
    print(f"Kept non-geometry misconceptions: {len(kept_df):,}")
    print(f"Removed geometry misconceptions: {len(mapping_df) - len(kept_df):,}")
    print(f"Saved curated misconception list to: {args.output_csv}")
    if args.compat_output_csv:
        print(f"Updated compatibility file: {args.compat_output_csv}")


if __name__ == "__main__":
    main()