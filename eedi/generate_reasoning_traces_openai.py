#!/usr/bin/env python3
"""Generate OpenAI reasoning traces for the filtered EEDI datasets.

This script fills the trace columns used by the downstream pipeline:
  - correct_answer_pairs_eedi.csv -> target_question_reasoning_trace
  - distractor_pairs_eedi.csv -> target_distractor_explained_traces

It is resumable by default: rows that already appear to contain generated
traces are skipped, and completed work is flushed back to disk after each
write batch and at the end of each dataset pass.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import pandas as pd

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The openai package is required. Install it with: pip install openai"
    ) from exc


import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prompt import (
    build_eedi_correct_answer_trace_prompt,
    build_eedi_distractor_trace_prompt,
    parse_first_distractor_answer,
)


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
        "--max-rows-per-task",
        type=int,
        default=None,
        help="Generate at most this many rows for each dataset in this run",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="0-based row offset to start scanning from in each dataset",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate traces even if a row already appears to be completed",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=10,
        help="Write progress back to disk after this many newly generated rows per dataset",
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


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


def build_correct_answer_messages(row: pd.Series) -> list[dict[str, str]]:
    prompt = build_eedi_correct_answer_trace_prompt(
        question=str(row["question"]),
        target_answer=str(row["target_answer"]),
    )
    return [{"role": "user", "content": prompt}]


def build_distractor_messages(row: pd.Series) -> tuple[list[dict[str, str]], str]:
    gold_distractor = parse_first_distractor_answer(row["target_distractor_answers"])
    prompt = build_eedi_distractor_trace_prompt(
        question=str(row["question"]),
        correct_answer=str(row["correct_answer"]),
        target_incorrect_answer=gold_distractor,
        misconception_name=str(row["misconception_name"]),
    )
    return ([{"role": "user", "content": prompt}], gold_distractor)


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


def needs_correct_trace(row: pd.Series) -> bool:
    trace = str(row.get("target_question_reasoning_trace", "") or "").strip()
    target = str(row.get("target_answer", "") or "").strip()
    return (not trace) or trace == target


def needs_distractor_trace(row: pd.Series) -> bool:
    trace = str(row.get("target_distractor_explained_traces", "") or "").strip()
    target = str(row.get("target_distractor_answers", "") or "").strip()
    return (not trace) or trace == target


def format_distractor_trace_value(trace: str) -> str:
    return json.dumps([trace], ensure_ascii=False)


def update_distractor_trace_columns(df: pd.DataFrame, row_idx: int, trace: str) -> None:
    trace_value = format_distractor_trace_value(trace)
    for column in [
        "distractor_reasoning_traces",
        "target_distractor_reasoning_traces",
        "distractor_explained_traces",
        "target_distractor_explained_traces",
    ]:
        if column in df.columns:
            df.at[row_idx, column] = trace_value


def print_correct_preview(row: pd.Series, trace: str) -> None:
    print("=" * 100)
    print(f"Correct-answer row {row.name} | question_id={row['question_id']}")
    print("-" * 100)
    print(f"question:\n{row['question']}")
    print()
    print(f"correct answer:\n{row['target_answer']}")
    print()
    print("generated reasoning trace:")
    print(trace)
    print()


def print_distractor_preview(row: pd.Series, target_distractor: str, trace: str) -> None:
    print("=" * 100)
    print(f"Distractor row {row.name} | question_id={row['question_id']}")
    print("-" * 100)
    print(f"question:\n{row['question']}")
    print()
    print(f"target distractor:\n{target_distractor}")
    print()
    print(f"misconception:\n{row['misconception_name']}")
    print()
    print("generated reasoning trace:")
    print(trace)
    print()


def maybe_flush_dataset(df: pd.DataFrame, path: Path, pending_writes: int, force: bool = False) -> int:
    if pending_writes <= 0:
        return 0
    if force or pending_writes > 0:
        write_csv_atomic(df, path)
        return 0
    return pending_writes


def process_correct_rows(args: argparse.Namespace, client: OpenAI) -> int:
    df = pd.read_csv(args.correct_csv)
    processed = 0
    pending_writes = 0
    for row_idx in range(args.start_row, len(df)):
        row = df.iloc[row_idx]
        if not args.overwrite and not needs_correct_trace(row):
            continue

        messages = build_correct_answer_messages(row)
        trace = call_openai(
            client,
            args.model,
            messages,
            args.max_output_tokens,
            args.temperature,
        )
        df.at[row_idx, "target_question_reasoning_trace"] = trace
        print_correct_preview(df.iloc[row_idx], trace)
        processed += 1
        pending_writes += 1
        if pending_writes >= args.write_batch_size:
            pending_writes = maybe_flush_dataset(df, args.correct_csv, pending_writes, force=True)
            print(
                f"Saved a correct-answer batch ending at row {row_idx} to {args.correct_csv}"
            )

        if args.max_rows_per_task is not None and processed >= args.max_rows_per_task:
            break

    pending_writes = maybe_flush_dataset(df, args.correct_csv, pending_writes, force=True)
    if processed:
        print(f"Saved final correct-answer progress to {args.correct_csv}")

    return processed


def process_distractor_rows(args: argparse.Namespace, client: OpenAI) -> int:
    df = pd.read_csv(args.distractor_csv)
    processed = 0
    pending_writes = 0
    for row_idx in range(args.start_row, len(df)):
        row = df.iloc[row_idx]
        if not args.overwrite and not needs_distractor_trace(row):
            continue

        messages, target_distractor = build_distractor_messages(row)
        trace = call_openai(
            client,
            args.model,
            messages,
            args.max_output_tokens,
            args.temperature,
        )
        update_distractor_trace_columns(df, row_idx, trace)
        print_distractor_preview(df.iloc[row_idx], target_distractor, trace)
        processed += 1
        pending_writes += 1
        if pending_writes >= args.write_batch_size:
            pending_writes = maybe_flush_dataset(df, args.distractor_csv, pending_writes, force=True)
            print(
                f"Saved a distractor batch ending at row {row_idx} to {args.distractor_csv}"
            )

        if args.max_rows_per_task is not None and processed >= args.max_rows_per_task:
            break

    pending_writes = maybe_flush_dataset(df, args.distractor_csv, pending_writes, force=True)
    if processed:
        print(f"Saved final distractor progress to {args.distractor_csv}")

    return processed


def main() -> None:
    args = parse_args()
    api_key = load_api_key(args)
    client = OpenAI(api_key=api_key)

    print(f"Model: {args.model}")
    print(f"Correct CSV: {args.correct_csv}")
    print(f"Distractor CSV: {args.distractor_csv}")
    print(f"Start row: {args.start_row}")
    print(f"Max rows per task: {args.max_rows_per_task}")
    print(f"Overwrite existing traces: {args.overwrite}")
    print(f"Write batch size: {args.write_batch_size}")
    print()

    correct_processed = process_correct_rows(args, client)
    distractor_processed = process_distractor_rows(args, client)

    print("=" * 100)
    print(f"Correct-answer rows processed this run: {correct_processed}")
    print(f"Distractor rows processed this run: {distractor_processed}")


if __name__ == "__main__":
    main()