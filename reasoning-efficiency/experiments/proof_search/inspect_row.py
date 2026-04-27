#!/usr/bin/env python3
"""Inspect a single row from a task CSV.

Usage:
    python inspect_row.py --task correct_answer --idx 42
    python inspect_row.py --task next_subquestion --idx 7 --split test
    python inspect_row.py --task distractor --idx 0
"""
import argparse
import json
import textwrap

import pandas as pd

TASK_CSV = {
    "correct_answer": "out/correct_answer_pairs.csv",
    "next_subquestion": "out/next_subquestion_pairs.csv",
    "distractor": "out/distractor_pairs.csv",
}
TASK_GOLD = {
    "correct_answer": "target_answer",
    "next_subquestion": "next_subquestion",
    "distractor": "target_distractor_answer",
}

W = 100  # wrap width

def hr(char="─"):
    print(char * W)

def section(title):
    hr("═")
    print(f"  {title}")
    hr("═")

def field(label, value, indent=2):
    prefix = " " * indent + f"{label}: "
    wrapped = textwrap.fill(str(value), width=W, initial_indent=prefix,
                            subsequent_indent=" " * (indent + len(label) + 2))
    print(wrapped)

def print_model_results(row, task):
    model_cols = [c for c in row.index if c not in {
        "example_idx", "problem_type", "complexity", "overlap_type", "problem",
        "question", "groundquery", "reasoning_trace", "answer", "depth", "width",
        "subquestions", "subquestions_by_node_id", "subquestion_tree", "trees",
        "pair_index", "solved_node_ids", "solved_steps", "target_node_id",
        "next_subquestion", "prompt", "target_question", "target_problem",
        "target_answer", "split",
        "correct_answer", "distractor_answer", "distractor_reasoning_trace",
        "misconception_type", "target_distractor_answer",
        "target_distractor_reasoning_trace",
    }]
    if not model_cols:
        print("  (no model result columns found)")
        return

    for col in sorted(model_cols):
        raw = row[col]
        if pd.isna(raw):
            print(f"  {col:<35} —  (train split, not inferred)")
            continue
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            print(f"  {col:<35} (unparseable: {raw})")
            continue

        b_pred, b_score = data["before"]
        a_pred, a_score = data["after"]
        gain = a_score - b_score
        gain_str = f"+{gain}" if gain > 0 else str(gain)
        status = "✓" if a_score == 1 else "✗"

        hr("·")
        print(f"  Model : {col}    [{status}]  before={b_score}  after={a_score}  gain={gain_str}")
        field("  Before pred", b_pred, indent=4)
        field("  After  pred", a_pred, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["correct_answer", "next_subquestion", "distractor"])
    parser.add_argument("--idx", required=True, type=int, help="Row number (0-based positional index)")
    parser.add_argument("--split", default=None, choices=["train", "test"],
                        help="Filter to a specific split before indexing")
    args = parser.parse_args()

    df = pd.read_csv(TASK_CSV[args.task])
    if args.split:
        df = df[df["split"] == args.split].reset_index(drop=True)

    if args.idx >= len(df):
        print(f"Index {args.idx} out of range (max {len(df)-1} for split={args.split or 'all'})")
        return

    row = df.iloc[args.idx]
    gold_col = TASK_GOLD[args.task]

    section(f"TASK: {args.task.upper()}  |  row={args.idx}  split={row.get('split','?')}")

    field("example_idx",   row.get("example_idx", "—"))
    field("pair_index",    row.get("pair_index", "—"))
    field("problem_type",  row.get("problem_type", ""))
    field("complexity",    row.get("complexity", ""))

    hr()
    print("  PROBLEM:")
    field("", row["problem"])

    if args.task == "distractor":
        hr()
        field("CORRECT ANSWER", row.get("correct_answer", "—"))
        hr()
        print("  REASONING TRACE (correct):")
        for line in str(row.get("reasoning_trace", "")).splitlines():
            print("   ", line)
        hr()
        field("MISCONCEPTION TYPE", row.get("misconception_type", "—"))
        hr()
        field("DISTRACTOR ANSWER", row.get("distractor_answer", "—"))
        hr()
        print("  DISTRACTOR REASONING TRACE:")
        for line in str(row.get("distractor_reasoning_trace", "")).splitlines():
            print("   ", line)

    hr()
    print("  PROMPT (sent to model):")
    for line in str(row["prompt"]).splitlines():
        print("   ", line)

    hr()
    field("GOLD (target)", row[gold_col])

    hr()
    print("  MODEL RESULTS (before → after fine-tuning):")
    print_model_results(row, args.task)
    hr("═")


if __name__ == "__main__":
    main()
