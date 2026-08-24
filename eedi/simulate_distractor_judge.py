#!/usr/bin/env python3
"""Simulate EEDI distractor judging on a few hand-crafted candidates.

This script samples a few rows from the processed EEDI distractor dataset,
constructs several made-up generations for each gold target, and runs the same
OpenAI semantic equivalence judge prompt used by the pipeline. It prints the
full judge prompt for each candidate so the behavior can be inspected directly.

Example:
    /cluster/home/tunguyen1/miniconda3/envs/llmgap/bin/python \
        eedi/simulate_distractor_judge.py \
        --num-examples 3 \
        --batch-size 8
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prompt import build_answer_equivalence_prompt, parse_first_distractor_answer
from openai_judge import parse_yes_no_judgment

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The openai package is required. Install it with: pip install openai"
    ) from exc


DEFAULT_DATA_CSV = (
    Path(__file__).resolve().parent / "data" / "processed" / "distractor_pairs_eedi.csv"
)
DEFAULT_API_KEY_FILE = Path(__file__).resolve().parent / "openai_api.txt"
DEFAULT_MODEL_NAME = os.environ.get("OPENAI_EEDI_JUDGE_MODEL", "gpt-4.1-mini")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=DEFAULT_DATA_CSV,
        help="Processed EEDI distractor CSV to sample from",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of gold distractor rows to inspect",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for row sampling",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="OpenAI judge model to call",
    )
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


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^(distractor\s*\d*\s*:|incorrect student answer:|answer:)\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def judge_candidate(client: OpenAI, model_name: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=256,
    )
    return (response.choices[0].message.content or "").strip()


def make_candidates(gold: str, correct_answer: str):
    gold = normalize_text(gold)
    return [
        ("equivalent_exact", gold),
        (
            "equivalent_reasoned",
            f"The student follows the misconception and ends with the distractor {gold}.",
        ),
        ("equivalent_prefixed", f"Answer: {gold}"),
        (
            "not_equivalent_correct",
            f"The student actually gets the correct answer {correct_answer}.",
        ),
        ("not_equivalent_generic", "The student picks None of these."),
        ("not_equivalent_modified", f"The student ends with {gold} + 1."),
    ]


def sample_rows(df: pd.DataFrame, num_examples: int, seed: int) -> pd.DataFrame:
    if num_examples >= len(df):
        return df.copy().reset_index(drop=True)
    return df.sample(n=num_examples, random_state=seed).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    api_key = load_api_key(args)
    df = pd.read_csv(args.data_csv)
    sampled = sample_rows(df, args.num_examples, args.seed)
    client = OpenAI(api_key=api_key)

    items = []
    for _, row in sampled.iterrows():
        gold = normalize_text(parse_first_distractor_answer(row["target_distractor_answers"]))
        correct_answer = normalize_text(row["correct_answer"])

        for label, candidate in make_candidates(gold, correct_answer):
            judge_prompt = build_answer_equivalence_prompt(
                row["prompt"],
                normalize_text(candidate),
                gold,
            )
            items.append(
                {
                    "question_id": row["question_id"],
                    "misconception_name": row["misconception_name"],
                    "gold": gold,
                    "candidate_type": label,
                    "candidate": candidate,
                    "prompt": judge_prompt,
                }
            )

    raw_outputs = [judge_candidate(client, args.model_name, item["prompt"]) for item in items]

    print(f"OpenAI judge model: {args.model_name}")
    print(f"Rows sampled: {len(sampled)}")
    print(f"Candidates judged: {len(items)}")
    print()

    current_question_id = None
    for item, raw_output in zip(items, raw_outputs):
        verdicts = re.findall(r"\b(yes|no)\b", raw_output, flags=re.IGNORECASE)
        verdict = bool(parse_yes_no_judgment(raw_output))
        if item["question_id"] != current_question_id:
            current_question_id = item["question_id"]
            print("=" * 100)
            print(f"question_id: {item['question_id']}")
            print(f"misconception: {item['misconception_name']}")
            print(f"gold distractor: {item['gold']}")
            print("-" * 100)

        print(f"[{item['candidate_type']}]")
        print(f"made-up generation: {item['candidate']}")
        print("judge prompt start")
        print(item["prompt"])
        print("judge prompt end")
        print("judge raw output start")
        print(raw_output)
        print("judge raw output end")
        print(f"judge raw output repr: {raw_output!r}")
        print(f"parsed verdict tokens: {verdicts}")
        print(f"judge verdict: {verdict}")
        print()


if __name__ == "__main__":
    main()