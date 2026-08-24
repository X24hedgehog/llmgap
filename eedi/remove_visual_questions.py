#!/usr/bin/env python3
"""Keep only EEDI rows whose misconceptions and question formats are allowed.

This script reads the full processed EEDI datasets and filters them down to the
question_ids whose distractor row uses an allowed misconception. The intended
workflow is:

1. Start from the full `*_with_visual.csv` datasets.
2. Build `misconceptions.csv` from the raw misconception mapping.
3. Keep only rows whose distractor misconception is in that curated list.
4. Drop question ids whose question text contains image markup (`![`).
5. Drop Tom/Katie argument-style questions that are closer to MCQ debate prompts.

The correct-answer CSV does not carry misconception ids, so filtering is driven
by the distractor CSV and then propagated to the aligned correct-answer CSV via
`question_id`.

Default outputs:
	- correct_answer_pairs_eedi.csv
	- distractor_pairs_eedi.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
DEFAULT_ALLOWED_MISCONCEPTIONS = DEFAULT_PROCESSED_DIR / "misconceptions.csv"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--processed-dir",
		type=Path,
		default=DEFAULT_PROCESSED_DIR,
		help="Directory containing the processed EEDI CSVs",
	)
	parser.add_argument(
		"--allowed-misconceptions",
		type=Path,
		default=DEFAULT_ALLOWED_MISCONCEPTIONS,
		help="CSV containing the allowed misconception ids",
	)
	parser.add_argument(
		"--correct-input",
		type=Path,
		default=None,
		help="Optional override for the full correct-answer CSV path",
	)
	parser.add_argument(
		"--distractor-input",
		type=Path,
		default=None,
		help="Optional override for the full distractor CSV path",
	)
	parser.add_argument(
		"--correct-output",
		type=Path,
		default=None,
		help="Optional override for the filtered correct-answer CSV path",
	)
	parser.add_argument(
		"--distractor-output",
		type=Path,
		default=None,
		help="Optional override for the filtered distractor CSV path",
	)
	return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
	correct_input = args.correct_input or (args.processed_dir / "correct_answer_pairs_eedi_with_visual.csv")
	distractor_input = args.distractor_input or (args.processed_dir / "distractor_pairs_eedi_with_visual.csv")
	correct_output = args.correct_output or (args.processed_dir / "correct_answer_pairs_eedi.csv")
	distractor_output = args.distractor_output or (args.processed_dir / "distractor_pairs_eedi.csv")
	return correct_input, distractor_input, correct_output, distractor_output


def load_allowed_misconception_ids(path: Path) -> set[int]:
	allowed_df = pd.read_csv(path)
	for column in ["MisconceptionId", "misconception_id"]:
		if column in allowed_df.columns:
			return set(allowed_df[column].astype(int).tolist())
	raise ValueError(
		f"Could not find a misconception id column in {path}. Expected MisconceptionId or misconception_id."
	)


def has_image_markup(question: str) -> bool:
	return "![" in str(question)


def is_tom_katie_argument_question(question: str) -> bool:
	text = str(question)
	text_lower = text.lower()
	return (
		"tom" in text_lower
		and "katie" in text_lower
		and "says" in text_lower
		and (
			"who is correct" in text_lower
			or "who do you agree with" in text_lower
		)
	)


def is_allowed_question(question: str) -> bool:
	return not has_image_markup(question) and not is_tom_katie_argument_question(question)


def main() -> None:
	args = parse_args()
	correct_input, distractor_input, correct_output, distractor_output = resolve_paths(args)
	allowed_misconception_ids = load_allowed_misconception_ids(args.allowed_misconceptions)

	correct_df = pd.read_csv(correct_input)
	distractor_df = pd.read_csv(distractor_input)

	allowed_distractor_df = distractor_df[
		distractor_df["misconception_id"].isin(allowed_misconception_ids)
	].copy()
	allowed_distractor_df = allowed_distractor_df[
		allowed_distractor_df["question"].map(is_allowed_question)
	].copy()
	allowed_question_ids = set(allowed_distractor_df["question_id"].tolist())

	filtered_correct_df = correct_df[
		correct_df["question_id"].isin(allowed_question_ids)
		& correct_df["question"].map(is_allowed_question)
	].reset_index(drop=True)
	filtered_distractor_df = allowed_distractor_df[
		allowed_distractor_df["question_id"].isin(set(filtered_correct_df["question_id"].tolist()))
	].reset_index(drop=True)

	correct_output.parent.mkdir(parents=True, exist_ok=True)
	distractor_output.parent.mkdir(parents=True, exist_ok=True)
	filtered_correct_df.to_csv(correct_output, index=False)
	filtered_distractor_df.to_csv(distractor_output, index=False)

	removed_question_ids = set(correct_df["question_id"].tolist()) - set(filtered_correct_df["question_id"].tolist())
	print("Filtered both datasets using the allowed misconception list.")
	print(f"Allowed misconception ids loaded: {len(allowed_misconception_ids):,}")
	print(f"Question ids removed: {len(removed_question_ids):,}")
	print(f"Correct-answer rows: {len(correct_df):,} -> {len(filtered_correct_df):,}")
	print(f"Distractor rows:     {len(distractor_df):,} -> {len(filtered_distractor_df):,}")
	print(f"Saved correct-answer CSV to: {correct_output}")
	print(f"Saved distractor CSV to:     {distractor_output}")
	print("Extra exclusions applied: image-marked questions and Tom/Katie argument questions")
	print(
		"Split counts after filtering (correct-answer):",
		filtered_correct_df["split"].value_counts().sort_index().to_dict(),
	)
	print(
		"Split counts after filtering (distractor):",
		filtered_distractor_df["split"].value_counts().sort_index().to_dict(),
	)


if __name__ == "__main__":
	main()
