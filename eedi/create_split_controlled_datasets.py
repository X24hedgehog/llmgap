#!/usr/bin/env python3
"""Create split-controlled EEDI datasets.

Rules are applied per misconception frequency in the distractor dataset, with
split tuple order: (train, validation, test).

Counts requested by the user:
  n=1  -> 1/0/0
  n=2  -> 1/0/1
  n=3  -> 1/1/1
  n=4  -> 2/1/1
  n=5  -> 3/1/1
  n>=6 -> [n - 2*(n//3), n//3, n//3]

The distractor dataset determines the split assignment. Because the processed
EEDI data has exactly one distractor row and one correct-answer row per
question_id, the same split is then copied to the matching correct-answer row.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"

DISTRACTOR_IN = PROCESSED / "distractor_pairs_eedi.csv"
CORRECT_IN = PROCESSED / "correct_answer_pairs_eedi.csv"

DISTRACTOR_OUT = PROCESSED / "distractor_pairs_eedi_split_control.csv"
CORRECT_OUT = PROCESSED / "correct_answer_pairs_eedi_split_control.csv"

SPLIT_ORDER = ("train", "validation", "test")


def target_counts(n_rows: int) -> tuple[int, int, int]:
    if n_rows == 1:
        return (1, 0, 0)
    if n_rows == 2:
        return (1, 0, 1)
    if n_rows == 3:
        return (1, 1, 1)
    if n_rows == 4:
        return (2, 1, 1)
    if n_rows == 5:
        return (3, 1, 1)
    third = n_rows // 3
    return (n_rows - 2 * third, third, third)


def sort_key(row: dict[str, str]) -> tuple[int, str]:
    question_id = int(row["question_id"])
    pair_index = row.get("pair_index", "")
    return (question_id, pair_index)


def assign_splits_by_misconception(rows: list[dict[str, str]]) -> dict[str, str]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["misconception_name"]].append(row)

    question_to_split = {}
    for misconception_name, group_rows in grouped.items():
        ordered = sorted(group_rows, key=sort_key)
        counts = target_counts(len(ordered))
        start = 0
        for split_name, split_count in zip(SPLIT_ORDER, counts):
            for row in ordered[start : start + split_count]:
                question_id = row["question_id"]
                if question_id in question_to_split:
                    raise ValueError(
                        f"question_id {question_id} received multiple split assignments"
                    )
                question_to_split[question_id] = split_name
            start += split_count

        if start != len(ordered):
            raise ValueError(
                f"Split accounting failed for misconception {misconception_name!r}"
            )

    return question_to_split


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], label: str) -> None:
    split_counter = Counter(row["split"] for row in rows)
    print(
        f"{label}: total={len(rows)} | "
        f"train={split_counter['train']} | "
        f"validation={split_counter['validation']} | "
        f"test={split_counter['test']}"
    )


def main() -> None:
    distractor_rows = list(csv.DictReader(DISTRACTOR_IN.open(newline="", encoding="utf-8")))
    correct_rows = list(csv.DictReader(CORRECT_IN.open(newline="", encoding="utf-8")))

    distractor_question_ids = [row["question_id"] for row in distractor_rows]
    if len(distractor_question_ids) != len(set(distractor_question_ids)):
        raise ValueError("Expected exactly one distractor row per question_id")

    correct_question_ids = {row["question_id"] for row in correct_rows}
    if correct_question_ids != set(distractor_question_ids):
        raise ValueError("Correct-answer and distractor question_id sets do not match")

    question_to_split = assign_splits_by_misconception(distractor_rows)

    distractor_out_rows = []
    for row in distractor_rows:
        out_row = dict(row)
        out_row["split"] = question_to_split[row["question_id"]]
        distractor_out_rows.append(out_row)

    correct_out_rows = []
    for row in correct_rows:
        out_row = dict(row)
        out_row["split"] = question_to_split[row["question_id"]]
        correct_out_rows.append(out_row)

    # Safety check: any misconception in validation/test must also appear in train.
    by_split = {
        split_name: Counter(
            row["misconception_name"] for row in distractor_out_rows if row["split"] == split_name
        )
        for split_name in SPLIT_ORDER
    }
    train_misconceptions = set(by_split["train"])
    for split_name in ("validation", "test"):
        missing = sorted(set(by_split[split_name]) - train_misconceptions)
        if missing:
            raise ValueError(
                f"Found misconceptions in {split_name} missing from train: {missing[:10]}"
            )

    write_csv(DISTRACTOR_OUT, distractor_out_rows, distractor_rows[0].keys())
    write_csv(CORRECT_OUT, correct_out_rows, correct_rows[0].keys())

    print(f"Wrote {DISTRACTOR_OUT}")
    print(f"Wrote {CORRECT_OUT}")
    summarize(distractor_out_rows, "distractor")
    summarize(correct_out_rows, "correct_answer")

    misconception_split_counts = {
        split_name: len(by_split[split_name]) for split_name in SPLIT_ORDER
    }
    print(
        "misconception coverage | "
        f"train={misconception_split_counts['train']} | "
        f"validation={misconception_split_counts['validation']} | "
        f"test={misconception_split_counts['test']}"
    )


if __name__ == "__main__":
    main()