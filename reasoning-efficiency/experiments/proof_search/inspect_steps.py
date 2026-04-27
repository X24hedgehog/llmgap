"""
inspect_steps.py

Pretty-prints rows from debug or final CSVs produced by prepare_gsm8k_v2.py.

Tasks:
  steps  -- steps-debug CSV (steps 1 & 2 output)
  ns     -- final next-subquestion CSV (full pipeline output)

Usage:
  python inspect_steps.py                          # steps task, 5 random rows
  python inspect_steps.py --task ns                # ns task, 5 random rows
  python inspect_steps.py --task ns --n 10
  python inspect_steps.py --task ns --index 0 4 17
  python inspect_steps.py --file out/my_file.csv   # explicit file
"""

import argparse
import json
import random
import re
import textwrap
from pathlib import Path

import pandas as pd

DEFAULT_FILES = {
    "steps": Path("out/gsm8k_v2_steps_debug.csv"),
    "ns":    Path("out/next_subquestion_pairs_gsm8k_v2.csv"),
    "ca":    Path("out/correct_answer_pairs_gsm8k.csv"),
}
GLOB_PATTERNS = {
    "steps": "gsm8k_v2_steps_debug*.csv",
    "ns":    "next_subquestion_pairs_gsm8k_v2*.csv",
    "ca":    "correct_answer_pairs_gsm8k*.csv",
}

WIDTH     = 100
SEP       = "─" * WIDTH
THICK_SEP = "═" * WIDTH


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(str(text), width=WIDTH,
                         initial_indent=indent, subsequent_indent=indent)


def print_tree(tree: dict) -> None:
    if not tree:
        print("    (empty tree)")
        return
    for nid, node in tree.items():
        children     = node.get("children", [])
        children_str = ", ".join(children) if children else "—"
        print(f"    {nid}  children=[{children_str}]")
        print(wrap(f"content: {node.get('content', '')}", indent="        "))


def print_steps(steps: list) -> None:
    if not steps:
        print("    (no steps parsed)")
        return
    for s in steps:
        print(f"    {s['step_id']}  result={s['result']}  expr={s['expression']}")
        print(wrap(f"explanation: {s['explanation']}", indent="        "))


# ── per-task row printers ─────────────────────────────────────────────────────

def inspect_row_steps(idx: int, row: pd.Series) -> None:
    print(f"\n{THICK_SEP}")
    print(f"  ROW {idx}  (split={row['split']})")
    print(THICK_SEP)

    print("\n  QUESTION:")
    print(wrap(row["question"]))

    print(f"\n{SEP}")
    print("  REASONING TRACE (raw):")
    for line in str(row["reasoning_trace"]).split("\n"):
        print(f"    {line}")

    print(f"\n{SEP}")
    steps = json.loads(row["steps_json"])
    print(f"  PARSED STEPS ({len(steps)} total):")
    print_steps(steps)

    print(f"\n{SEP}")
    tree = json.loads(row["tree_json"])
    print(f"  DEPENDENCY TREE ({len(tree)} nodes):")
    print_tree(tree)
    print()


def find_node_id_by_content(tree: dict, content: str) -> str | None:
    """Return the node id whose content matches the given string."""
    for nid, node in tree.items():
        if isinstance(node, dict) and node.get("content", "").strip() == content.strip():
            return nid
    return None


def find_context_node_ids(tree: dict, prompt: str) -> list[str]:
    """
    Extract context node ids by matching each tree node's content against the
    context section of the prompt (between 'questions: ' and 'Next subquestion:').
    """
    m = re.search(r'questions:\s*(.*?)\s*Next subquestion:', prompt, re.DOTALL)
    if not m:
        return []
    context_block = m.group(1).strip()
    matched = []
    for nid, node in tree.items():
        if not isinstance(node, dict):
            continue
        content = node.get("content", "").strip().rstrip("?")
        if content and content in context_block:
            matched.append(nid)
    return matched


def inspect_row_ns(idx: int, row: pd.Series) -> None:
    print(f"\n{THICK_SEP}")
    print(f"  ROW {idx}  (split={row['split']})")
    print(THICK_SEP)

    print("\n  QUESTION:")
    print(wrap(row["question"]))

    print(f"\n{SEP}")
    print("  REASONING TRACE (raw):")
    for line in str(row["reasoning_trace"]).split("\n"):
        print(f"    {line}")

    print(f"\n{SEP}")
    tree_raw = row.get("tree", "{}")
    tree = json.loads(tree_raw) if isinstance(tree_raw, str) else {}
    print(f"  TREE ({len(tree)} nodes):")
    print_tree(tree)

    print(f"\n{SEP}")
    target_text = str(row.get("next_subquestion", ""))
    target_nid  = find_node_id_by_content(tree, target_text) if tree else None
    target_label = f"[{target_nid}]" if target_nid else "[id unknown]"
    print(f"  TARGET NEXT SUBQUESTION  {target_label}:")
    print(wrap(target_text))

    print(f"\n{SEP}")
    prompt = str(row.get("prompt", ""))
    context_nids = find_context_node_ids(tree, prompt) if tree else []
    context_label = f"[{', '.join(context_nids)}]" if context_nids else "[none / no context]"
    print(f"  PROMPT  (context nodes: {context_label}):")
    for line in prompt.split("\n"):
        print(f"    {line}")
    print()


def inspect_row_ca(idx: int, row: pd.Series) -> None:
    print(f"\n{THICK_SEP}")
    print(f"  ROW {idx}  (split={row['split']})")
    print(THICK_SEP)

    print("\n  QUESTION:")
    print(wrap(row["question"]))

    print(f"\n{SEP}")
    print("  PROMPT:")
    for line in str(row.get("prompt", "")).split("\n"):
        print(f"    {line}")

    print(f"\n{SEP}")
    print(f"  TARGET ANSWER:  {row.get('target_answer', '')}")

    print(f"\n{SEP}")
    print("  TARGET REASONING TRACE (CoT ground truth):")
    for line in str(row.get("target_question_reasoning_trace", "")).split("\n"):
        print(f"    {line}")
    print()


# ── question quality stats ────────────────────────────────────────────────────

QUESTION_STARTERS = re.compile(
    r'^(what|how|who|when|where|which|why|is|are|does|do|did|can|could|if)',
    re.IGNORECASE,
)


def compute_question_quality(df: pd.DataFrame, label: str) -> None:
    """
    For every node content in the 'tree' column, check whether it looks like a
    proper question (starts with a question word and ends with '?').
    Prints per-file statistics.
    """
    total_nodes   = 0
    good_nodes    = 0
    bad_examples  = []   # collect a few bad ones for display

    for _, row in df.iterrows():
        tree_raw = row.get("tree", "{}")
        try:
            tree = json.loads(tree_raw) if isinstance(tree_raw, str) else {}
        except Exception:
            continue
        for nid, node in tree.items():
            if not isinstance(node, dict):
                continue
            content = str(node.get("content", "")).strip()
            if not content:
                continue
            total_nodes += 1
            is_good = content.endswith("?") and bool(QUESTION_STARTERS.match(content))
            if is_good:
                good_nodes += 1
            elif len(bad_examples) < 5:
                bad_examples.append((nid, content))

    if total_nodes == 0:
        print(f"  [{label}] No tree nodes found.")
        return

    pct = 100.0 * good_nodes / total_nodes
    print(f"\n  [{label}]")
    print(f"    Total tree nodes : {total_nodes:,}")
    print(f"    Well-formed      : {good_nodes:,}  ({pct:.1f}%)")
    print(f"    Malformed        : {total_nodes - good_nodes:,}  ({100-pct:.1f}%)")
    if bad_examples:
        print(f"    Sample malformed nodes:")
        for nid, content in bad_examples:
            short = content[:120] + ("…" if len(content) > 120 else "")
            print(f"      [{nid}] {short}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  default="steps", choices=["steps", "ns", "ca"],
                        help="Which CSV to inspect: 'steps' (debug), 'ns' (next subquestion), or 'ca' (correct answer)")
    parser.add_argument("--file",  default=None,
                        help="Explicit CSV path (overrides --task default)")
    parser.add_argument("--n",     type=int, default=5,
                        help="Number of random rows to show (default: %(default)s)")
    parser.add_argument("--index", type=int, nargs="+", default=None,
                        help="Specific row indices to show (overrides --n)")
    parser.add_argument("--seed",  type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--stats", action="store_true",
                        help="Print question-quality statistics and exit (no row inspection)")
    parser.add_argument("--stats-files", nargs="+", default=None, metavar="CSV",
                        help="Compare question quality across multiple CSV files")
    args = parser.parse_args()

    # ── stats-only mode ───────────────────────────────────────────────────────
    if args.stats or args.stats_files:
        files = args.stats_files or []
        if not files:
            # default: compare v1 and v2_fixed
            files = [
                "out/next_subquestion_pairs_gsm8k.csv",
                "out/next_subquestion_pairs_gsm8k_v2_fixed.csv",
            ]
        print(f"\n{'═'*WIDTH}")
        print("  QUESTION QUALITY STATISTICS")
        print(f"{'═'*WIDTH}")
        for f in files:
            p = Path(f)
            if not p.exists():
                print(f"\n  [{f}] NOT FOUND — skipping")
                continue
            d = pd.read_csv(p)
            compute_question_quality(d, str(p))
        return

    # ── normal row-inspection mode ────────────────────────────────────────────
    if args.file:
        path = Path(args.file)
    else:
        path = DEFAULT_FILES[args.task]

    if not path.exists():
        candidates = sorted(Path("out").glob(GLOB_PATTERNS[args.task]))
        if not candidates:
            raise FileNotFoundError(
                f"{path} not found. "
                + ("Run: python prepare_gsm8k_v2.py --steps-only"
                   if args.task == "steps"
                   else "Run: python prepare_gsm8k_v2.py (or sbatch)")
            )
        path = candidates[-1]
        print(f"  (using {path})")

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")

    if args.index is not None:
        indices = args.index
    else:
        if args.seed is not None:
            random.seed(args.seed)
        n = min(args.n, len(df))
        indices = sorted(random.sample(range(len(df)), n))

    if args.task == "steps":
        inspect_fn = inspect_row_steps
    elif args.task == "ca":
        inspect_fn = inspect_row_ca
    else:
        inspect_fn = inspect_row_ns

    print(f"Inspecting rows: {indices}")
    for idx in indices:
        if idx < 0 or idx >= len(df):
            print(f"  [skip] index {idx} out of range (0–{len(df)-1})")
            continue
        inspect_fn(idx, df.iloc[idx])


if __name__ == "__main__":
    main()
