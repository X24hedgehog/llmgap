#!/usr/bin/env python3
"""Prepare EEDI train splits in the same CSV format used by the llmgap pipeline.

This script converts the labeled EEDI competition training data into two CSVs:
1. correct_answer_pairs_eedi.csv
2. distractor_pairs_eedi.csv

Design choices:
- We split at the question level to avoid leakage across distractors from the same item.
- The correct-answer task predicts the correct answer text for a multiple-choice item.
- The distractor task predicts one randomly sampled labeled distractor answer per
    problem, conditioned on the question, the correct answer, and the misconception
    label.
- Because EEDI does not provide reasoning traces, the target text fields are filled
  with answer-option text so they remain compatible with the existing pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


DEFAULT_RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data" / "processed"

ANSWER_LETTERS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing train.csv and misconception_mapping.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where processed CSVs will be written",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Question-level fraction reserved for the held-out test split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for question-level train/test splitting",
    )
    return parser.parse_args()


def _normalize_multiline(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text).strip().splitlines()).strip()


def _build_correct_prompt(row: pd.Series) -> str:
    question = _normalize_multiline(row["QuestionText"])
    options = [
        f"A: {_normalize_multiline(row['AnswerAText'])}",
        f"B: {_normalize_multiline(row['AnswerBText'])}",
        f"C: {_normalize_multiline(row['AnswerCText'])}",
        f"D: {_normalize_multiline(row['AnswerDText'])}",
    ]
    return (
        "You will be given a multiple-choice math question. "
        "Return only the correct answer text and nothing else.\n\n"
        f"Subject: {row['SubjectName']}\n"
        f"Construct: {row['ConstructName']}\n"
        f"Question: {question}\n"
        "Options:\n"
        + "\n".join(options)
    )


def _build_distractor_prompt(row: pd.Series, correct_answer_text: str, misconception_name: str) -> str:
    question = _normalize_multiline(row["QuestionText"])
    return (
        "You will be given a math question along with the correct answer and a specific "
        "student misconception. Please generate 1 incorrect distractor answer for the "
        "question to be used as a multiple-choice option in a multiple-choice exam. "
        "The distractor should be plausible for a student with the given misconception. "
        "Output only the distractor answer text.\n\n"
        f"Question: {question}\n"
        f"Answer: {correct_answer_text}\n"
        f"Student Error: {misconception_name}"
    )


def _split_questions(train_df: pd.DataFrame, test_frac: float, seed: int) -> dict[int, str]:
    qids = pd.Series(train_df["QuestionId"].drop_duplicates().tolist())
    qids = qids.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(qids) * (1 - test_frac))
    train_ids = set(qids.iloc[:split_idx].tolist())
    return {qid: ("train" if qid in train_ids else "test") for qid in qids.tolist()}


def build_correct_answer_df(train_df: pd.DataFrame, split_map: dict[int, str]) -> pd.DataFrame:
    rows = []
    for idx, row in train_df.iterrows():
        correct_letter = row["CorrectAnswer"]
        correct_answer_text = row[f"Answer{correct_letter}Text"]
        rows.append(
            {
                "example_idx": idx,
                "question_id": row["QuestionId"],
                "construct_id": row["ConstructId"],
                "construct_name": row["ConstructName"],
                "subject_id": row["SubjectId"],
                "subject_name": row["SubjectName"],
                "question": row["QuestionText"],
                "correct_answer_letter": correct_letter,
                "correct_answer": correct_answer_text,
                "answer_a_text": row["AnswerAText"],
                "answer_b_text": row["AnswerBText"],
                "answer_c_text": row["AnswerCText"],
                "answer_d_text": row["AnswerDText"],
                "prompt_style": "eedi_correct_answer",
                "prompt": _build_correct_prompt(row),
                "target_answer": correct_answer_text,
                "target_question_reasoning_trace": correct_answer_text,
                "split": split_map[row["QuestionId"]],
            }
        )
    return pd.DataFrame(rows)


def build_distractor_df(
    train_df: pd.DataFrame,
    split_map: dict[int, str],
    misconception_map: dict[int, str],
    seed: int,
) -> pd.DataFrame:
    rows = []
    pair_index = 0
    rng = random.Random(seed)
    for _, row in train_df.iterrows():
        correct_letter = row["CorrectAnswer"]
        correct_answer_text = row[f"Answer{correct_letter}Text"]
        candidates = []

        for answer_letter in ANSWER_LETTERS:
            if answer_letter == correct_letter:
                continue

            misconception_id = row[f"Misconception{answer_letter}Id"]
            if pd.isna(misconception_id):
                continue
            misconception_id = int(misconception_id)
            misconception_name = misconception_map.get(misconception_id, f"Misconception {misconception_id}")
            distractor_text = row[f"Answer{answer_letter}Text"]

            candidates.append(
                {
                    "answer_letter": answer_letter,
                    "distractor_text": distractor_text,
                    "misconception_id": misconception_id,
                    "misconception_name": misconception_name,
                }
            )

        if not candidates:
            continue

        selected = rng.choice(candidates)
        target_list = json.dumps([selected["distractor_text"]], ensure_ascii=False)

        rows.append(
            {
                "pair_index": pair_index,
                "question_id": row["QuestionId"],
                "answer_letter": selected["answer_letter"],
                "construct_id": row["ConstructId"],
                "construct_name": row["ConstructName"],
                "subject_id": row["SubjectId"],
                "subject_name": row["SubjectName"],
                "problem": row["QuestionText"],
                "question": row["QuestionText"],
                "correct_answer_letter": correct_letter,
                "correct_answer": correct_answer_text,
                "distractor_text": selected["distractor_text"],
                "misconception_id": selected["misconception_id"],
                "misconception_name": selected["misconception_name"],
                "misconception_type": selected["misconception_name"],
                "prompt_style": "eedi_distractor",
                "prompt": _build_distractor_prompt(
                    row,
                    correct_answer_text,
                    selected["misconception_name"],
                ),
                "distractor_answers": target_list,
                "target_distractor_answers": target_list,
                "distractor_reasoning_traces": target_list,
                "target_distractor_reasoning_traces": target_list,
                "distractor_explained_traces": target_list,
                "target_distractor_explained_traces": target_list,
                "split": split_map[row["QuestionId"]],
            }
        )
        pair_index += 1

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    train_path = args.raw_dir / "train.csv"
    misconception_path = args.raw_dir / "misconception_mapping.csv"

    if not train_path.exists() or not misconception_path.exists():
        raise SystemExit(
            f"Expected to find {train_path} and {misconception_path}. "
            "Run download_and_inspect_eedi.py first."
        )

    train_df = pd.read_csv(train_path)
    misconception_df = pd.read_csv(misconception_path)
    misconception_map = dict(
        zip(misconception_df["MisconceptionId"], misconception_df["MisconceptionName"])
    )

    split_map = _split_questions(train_df, test_frac=args.test_frac, seed=args.seed)
    correct_df = build_correct_answer_df(train_df, split_map)
    distractor_df = build_distractor_df(
        train_df,
        split_map,
        misconception_map,
        seed=args.seed,
    )

    # Align with distractor coverage, as in the synthetic setting.
    valid_qids = set(distractor_df["question_id"].unique())
    correct_df = correct_df[correct_df["question_id"].isin(valid_qids)].reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    correct_path = args.out_dir / "correct_answer_pairs_eedi.csv"
    distractor_path = args.out_dir / "distractor_pairs_eedi.csv"

    correct_df.to_csv(correct_path, index=False)
    distractor_df.to_csv(distractor_path, index=False)

    print("Saved EEDI datasets:")
    print(f"  correct_answer: {correct_path} ({len(correct_df):,} rows)")
    print(f"  distractor:     {distractor_path} ({len(distractor_df):,} rows)")
    print(
        "  split counts (correct_answer):",
        correct_df["split"].value_counts().sort_index().to_dict(),
    )
    print(
        "  split counts (distractor):",
        distractor_df["split"].value_counts().sort_index().to_dict(),
    )


if __name__ == "__main__":
    main()