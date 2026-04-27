"""
inspect_sample.py

Pretty-print the generated sample CSVs for quality inspection.

Usage:
  python inspect_sample.py                   # print both CSVs, all rows
  python inspect_sample.py --task ca         # only correct_answer
  python inspect_sample.py --task ns         # only next_subquestion
  python inspect_sample.py --index 3         # only row 3 (0-based)
  python inspect_sample.py --task ns --tree  # also print the full tree JSON
"""

import argparse
import json
from pathlib import Path

import pandas as pd

OUT_DIR = Path("out")
CA_CSV  = OUT_DIR / "correct_answer_pairs_gsm8k.csv"
NS_CSV  = OUT_DIR / "next_subquestion_pairs_gsm8k.csv"

SEP  = "─" * 80
SEP2 = "═" * 80


def fmt(label: str, value: str, width: int = 78) -> str:
    lines = str(value).splitlines()
    indented = ("\n" + " " * (len(label) + 2)).join(lines)
    return f"  {label}: {indented}"


def print_ca_row(i: int, row: pd.Series) -> None:
    print(f"\n{SEP2}")
    print(f"  [CA] Row {i}   split={row.get('split', '?')}")
    print(SEP2)
    print(fmt("QUESTION", row["question"]))
    print()
    print(fmt("TARGET_ANSWER", row["target_answer"]))
    print()
    print(fmt("REASONING_TRACE", row["reasoning_trace"]))
    print()
    print(fmt("TARGET_REASONING_TRACE", row["target_question_reasoning_trace"]))


def print_ns_row(i: int, row: pd.Series, show_tree: bool = False) -> None:
    print(f"\n{SEP2}")
    print(f"  [NS] Row {i}   split={row.get('split', '?')}")
    print(SEP2)
    print(fmt("QUESTION", row["question"]))
    print()
    print(fmt("REASONING_TRACE", row.get("reasoning_trace", "(not available)")))
    print()
    print(fmt("NEXT_SUBQUESTION (target)", row["next_subquestion"]))
    print()
    print(fmt("PROMPT (what model sees)", row["prompt"]))
    if show_tree:
        print()
        try:
            tree = json.loads(row["tree"])
            pretty = json.dumps(tree, indent=2)
        except Exception:
            pretty = str(row["tree"])
        print(fmt("TREE", pretty))


def print_csv(path: Path, task: str, indices, show_tree: bool) -> None:
    if not path.exists():
        print(f"[skip] {path} not found.")
        return
    df = pd.read_csv(path)
    print(f"\n{SEP}")
    print(f"  File : {path}   ({len(df)} rows)")
    print(SEP)
    rows_to_show = [df.iloc[i] for i in indices if i < len(df)] if indices else [row for _, row in df.iterrows()]
    idx_list     = indices if indices else list(range(len(df)))
    for i, row in zip(idx_list, rows_to_show):
        if task == "ca":
            print_ca_row(i, row)
        else:
            print_ns_row(i, row, show_tree=show_tree)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  choices=["ca", "ns", "both"], default="both",
                        help="Which CSV to inspect: ca, ns, or both (default: both)")
    parser.add_argument("--index", type=int, nargs="+", default=None,
                        help="Row indices to show (0-based). Default: all rows.")
    parser.add_argument("--tree",  action="store_true",
                        help="Also print full tree JSON for NS rows")
    args = parser.parse_args()

    if args.task in ("ca", "both"):
        print_csv(CA_CSV,  "ca", args.index, args.tree)
    if args.task in ("ns", "both"):
        print_csv(NS_CSV,  "ns", args.index, args.tree)


if __name__ == "__main__":
    main()
