#!/usr/bin/env python3
"""Smoke-test OpenAI reasoning-trace generation for EEDI tasks.

This script samples a few datapoints from the processed EEDI correct-answer and
distractor datasets, sends them to an OpenAI model, and prints the question,
gold answer or distractor target, misconception when relevant, and the
generated reasoning trace.

API key lookup order:
        1. --api-key
        2. --api-key-file (defaults to eedi/openai_api.txt)
        3. OPENAI_API_KEY

Example:
  python eedi/test_openai_reasoning_generation.py \
      --model gpt-5.4 \
      --num-examples-per-task 2
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

import pandas as pd

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The openai package is required. Install it with: pip install openai"
    ) from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_CORRECT_CSV = ROOT / "data" / "processed" / "correct_answer_pairs_eedi.csv"
DEFAULT_DISTRACTOR_CSV = ROOT / "data" / "processed" / "distractor_pairs_eedi.csv"
DEFAULT_API_KEY_FILE = ROOT / "openai_api.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.4", help="OpenAI model name")
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. If omitted, the script will try a local key file.",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=DEFAULT_API_KEY_FILE,
        help="Path to a file containing only the OpenAI API key",
    )
    parser.add_argument(
        "--correct-csv",
        type=Path,
        default=DEFAULT_CORRECT_CSV,
        help="Processed correct-answer CSV",
    )
    parser.add_argument(
        "--distractor-csv",
        type=Path,
        default=DEFAULT_DISTRACTOR_CSV,
        help="Processed distractor CSV",
    )
    parser.add_argument(
        "--num-examples-per-task",
        type=int,
        default=2,
        help="How many rows to sample from each task",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=14,
        help="Random seed for sampling rows",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=700,
        help="Maximum completion tokens for the generated reasoning trace",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation",
    )
    return parser.parse_args()


def load_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()

    if args.api_key_file.exists():
        return args.api_key_file.read_text(encoding="utf-8").strip()

    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    raise SystemExit(
        "No OpenAI API key found. Put your key in eedi/openai_api.txt, "
        "or pass --api-key, or set OPENAI_API_KEY."
    )


def sample_rows(df: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count >= len(df):
        return df.copy().reset_index(drop=True)
    return df.sample(n=count, random_state=seed).reset_index(drop=True)


def build_correct_answer_messages(row: pd.Series) -> list[dict[str, str]]:
    developer_message = (
        "You are writing a gold reasoning trace for a math tutoring dataset. "
        "You will be given: a math question; the known correct answer. "
        "Your job: produce a concise but clear step-by-step reasoning trace that "
        "solves the question; use the known correct answer as ground truth; end "
        "with a final sentence that states the correct answer exactly; return only "
        "the reasoning trace."
    )
    user_message = (
        f"Math question:\n{row['question']}\n\n"
        f"Known correct answer: {row['target_answer']}"
    )
    return [
        {"role": "developer", "content": developer_message},
        {"role": "user", "content": user_message},
    ]


def build_distractor_messages(row: pd.Series) -> tuple[list[dict[str, str]], str]:
    gold_values = ast.literal_eval(row["target_distractor_answers"])
    if not isinstance(gold_values, list):
        gold_values = [gold_values]
    gold_distractor = gold_values[0]
    developer_message = (
        "You are writing a gold reasoning trace for a student-misconception dataset. "
        "You will be given: a math question; the known correct answer; a target "
        "incorrect answer; the student's misconception. Your job: produce a concise "
        "but clear step-by-step incorrect reasoning trace that a student with this "
        "misconception could plausibly follow; the reasoning should be coherent with "
        "the misconception; it must end at the given target incorrect answer and not "
        "at the correct answer; return only the reasoning trace."
    )
    user_message = (
        f"Math question:\n{row['question']}\n\n"
        f"Known correct answer: {row['correct_answer']}\n"
        f"Target incorrect answer: {gold_distractor}\n"
        f"Student misconception: {row['misconception_name']}"
    )
    return (
        [
            {"role": "developer", "content": developer_message},
            {"role": "user", "content": user_message},
        ],
        str(gold_distractor),
    )


def call_openai(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_output_tokens,
        temperature=temperature,
    )

    return (response.choices[0].message.content or "").strip()


def print_example(
    header: str,
    row: pd.Series,
    gold_answer: str,
    generated_trace: str,
    misconception: str | None = None,
) -> None:
    print("=" * 100)
    print(header)
    print("-" * 100)
    print(f"question_id: {row['question_id']}")
    print(f"question:\n{row['question']}")
    print()
    if misconception is not None:
        print(f"target distractor:\n{gold_answer}")
        print()
        print(f"misconception:\n{misconception}")
    else:
        print(f"correct answer:\n{gold_answer}")
    print()
    print("generated reasoning trace:")
    print(generated_trace)
    print()


def main() -> None:
    args = parse_args()
    api_key = load_api_key(args)

    correct_df = pd.read_csv(args.correct_csv)
    distractor_df = pd.read_csv(args.distractor_csv)

    correct_rows = sample_rows(correct_df, args.num_examples_per_task, args.seed)
    distractor_rows = sample_rows(distractor_df, args.num_examples_per_task, args.seed)

    client = OpenAI(api_key=api_key)

    print(f"Model: {args.model}")
    print(f"Correct-answer samples: {len(correct_rows)}")
    print(f"Distractor samples: {len(distractor_rows)}")
    print()

    for _, row in correct_rows.iterrows():
        messages = build_correct_answer_messages(row)
        generated_trace = call_openai(
            client,
            args.model,
            messages,
            args.max_output_tokens,
            args.temperature,
        )
        print_example(
            "Correct Answer Task",
            row,
            str(row["target_answer"]),
            generated_trace,
        )

    for _, row in distractor_rows.iterrows():
        messages, gold_distractor = build_distractor_messages(row)
        generated_trace = call_openai(
            client,
            args.model,
            messages,
            args.max_output_tokens,
            args.temperature,
        )
        print_example(
            "Distractor Task",
            row,
            gold_distractor,
            generated_trace,
            misconception=str(row["misconception_name"]),
        )


if __name__ == "__main__":
    main()