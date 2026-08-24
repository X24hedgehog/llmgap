#!/usr/bin/env python3
"""
Fine-tune frontier/API models for the three tasks:
  - correct_answer
  - next_subquestion
  - distractor

This script is intentionally separate from finetune.py.

Supported:
  - OpenAI fine-tuning API
  - OpenAI-compatible providers ONLY if they implement:
      /v1/files
      /v1/fine_tuning/jobs

Typical usage:

python finetune_frontier.py \
  --provider openai \
  --base-model gpt-4.1-mini \
  --task correct_answer \
  --train-csv out/correct_answer_pairs_gsm8k.csv \
  --out-dir out/frontier_ft/gpt41mini_correct_answer

After the job succeeds, this script writes:
  out-dir/fine_tuned_model.txt

You then pass that path to run_inference_frontier.py with --mode after.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Prompt/response formatting copied conceptually from your existing scripts
# ──────────────────────────────────────────────────────────────────────────────

CORRECT_ANSWER_INSTRUCTION = "\n\nSolve this problem step by step."

DISTRACTOR_PROMPT_BY_TYPE = {
    "ContCompContMisconceptionIncons": (
        "You are an expert math educator generating a distractor based on "
        "the Comparison Inconsistency misconception.\n\n"
        "When a problem says \"A has N more than B\", the correct way to find B "
        "is B = A - N. A student with this misconception FLIPS the operation: "
        "they compute B = A + N instead. Similarly, \"fewer than\" → they subtract "
        "instead of add, \"times as many\" → they multiply instead of "
        "divide, etc.\n\n"
        "Your job is to simulate this mistake and produce a single distractor "
        "answer for the following math question.\n\n"
        "Question: {problem}\n"
        "Correct answer: {correct_answer}\n\n"
        "Think step by step, flipping one comparison operation to arrive at "
        "a wrong answer."
    ),
    "ContTransferContMisconceptionIncons": (
        "You are an expert math educator generating a distractor based on "
        "the Transfer Inconsistency misconception.\n\n"
        "When a problem says \"A gives B N items\", the correct effect on A "
        "is: A loses N (subtract). A student with this misconception FLIPS "
        "the operation: they add instead of subtract, or vice versa, "
        "confusing the direction of the transfer.\n\n"
        "Your job is to simulate this mistake and produce a single distractor "
        "answer for the following math question.\n\n"
        "Question: {problem}\n"
        "Correct answer: {correct_answer}\n\n"
        "Think step by step, flipping one transfer operation to arrive at "
        "a wrong answer."
    ),
}


def _safe_get(row: pd.Series, key: str, default: str = "") -> str:
    if key not in row or pd.isna(row[key]):
        return default
    return str(row[key])


def build_prompt(row: pd.Series, task: str) -> str:
    """
    Build prompt for one example.
    This mirrors the logic in your local SFT/inference scripts.
    """
    prompt_style = _safe_get(row, "prompt_style")

    if prompt_style == "eedi_correct_answer":
        return f"Question: {_safe_get(row, 'question').strip()}" + CORRECT_ANSWER_INSTRUCTION

    if prompt_style == "eedi_distractor":
        return (
            "You are solving a math question as a student with the following "
            f"misconception: {_safe_get(row, 'misconception_name').strip()}\n\n"
            f"Question: {_safe_get(row, 'question').strip()}\n\n"
            "Think step by step and give the incorrect answer this student "
            "would produce."
        )

    if task == "correct_answer":
        if "prompt" in row and not pd.isna(row["prompt"]):
            return str(row["prompt"]).rstrip() + CORRECT_ANSWER_INSTRUCTION
        if "question" in row and not pd.isna(row["question"]):
            return f"Question: {str(row['question']).strip()}" + CORRECT_ANSWER_INSTRUCTION
        raise ValueError("correct_answer row has neither prompt nor question")

    if task == "next_subquestion":
        if "prompt" not in row or pd.isna(row["prompt"]):
            raise ValueError("next_subquestion row requires prompt")
        return str(row["prompt"]).rstrip()

    if task == "distractor":
        mtype = _safe_get(row, "misconception_type")
        if mtype in DISTRACTOR_PROMPT_BY_TYPE:
            return DISTRACTOR_PROMPT_BY_TYPE[mtype].format(
                problem=_safe_get(row, "problem"),
                correct_answer=_safe_get(row, "correct_answer"),
            )

        if "prompt" in row and not pd.isna(row["prompt"]):
            return str(row["prompt"]).rstrip() + (
                "\n\nIncorrect Answer: Let's think step by step."
            )

        if "question" in row and not pd.isna(row["question"]):
            return (
                f"Question: {str(row['question']).strip()}\n\n"
                "Generate one plausible incorrect answer. Think step by step."
            )

        raise ValueError("distractor row has no usable prompt")

    raise ValueError(f"Unknown task: {task}")


def build_response(row: pd.Series, task: str) -> str:
    """
    Build assistant target for fine-tuning.
    """
    if task == "correct_answer":
        if "target_question_reasoning_trace" in row and not pd.isna(row["target_question_reasoning_trace"]):
            return str(row["target_question_reasoning_trace"]).strip()
        if "reasoning_trace" in row and not pd.isna(row["reasoning_trace"]):
            return str(row["reasoning_trace"]).strip()
        if "target_answer" in row and not pd.isna(row["target_answer"]):
            return f"#### The final answer is {str(row['target_answer']).strip()}"
        raise ValueError("correct_answer row has no target response")

    if task == "next_subquestion":
        if "next_subquestion" in row and not pd.isna(row["next_subquestion"]):
            return str(row["next_subquestion"]).strip()
        raise ValueError("next_subquestion row requires next_subquestion")

    if task == "distractor":
        if "target_distractor_explained_traces" in row and not pd.isna(row["target_distractor_explained_traces"]):
            traces = json.loads(str(row["target_distractor_explained_traces"]))
            if isinstance(traces, list) and traces:
                return str(random.choice(traces)).strip()
            return str(traces).strip()

        if "target_distractor_answers" in row and not pd.isna(row["target_distractor_answers"]):
            values = json.loads(str(row["target_distractor_answers"]))
            if isinstance(values, list) and values:
                return f"The plausible incorrect answer is {values[0]}."
            return f"The plausible incorrect answer is {values}."

        raise ValueError("distractor row has no target response")

    raise ValueError(f"Unknown task: {task}")


# ──────────────────────────────────────────────────────────────────────────────
# JSONL construction
# ──────────────────────────────────────────────────────────────────────────────

def make_chat_example(prompt: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def write_chat_jsonl(df: pd.DataFrame, task: str, out_jsonl: str) -> int:
    n_written = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            try:
                prompt = build_prompt(row, task)
                response = build_response(row, task)

                if not prompt.strip() or not response.strip():
                    continue

                item = make_chat_example(prompt, response)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_written += 1

            except Exception as e:
                print(f"[WARN] skipping row {idx}: {e}")

    return n_written


def load_training_rows(args) -> pd.DataFrame:
    df = pd.read_csv(args.train_csv)

    if "split" in df.columns:
        df = df[df["split"] == "train"].reset_index(drop=True)

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    if args.max_rows is not None:
        df = df.head(args.max_rows).reset_index(drop=True)

    return df


def split_train_val(df: pd.DataFrame, val_ratio: float, seed: int):
    if val_ratio <= 0:
        return df, None

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(df) * val_ratio))
    val_df = df.iloc[:n_val].reset_index(drop=True)
    train_df = df.iloc[n_val:].reset_index(drop=True)
    return train_df, val_df


# ──────────────────────────────────────────────────────────────────────────────
# Provider fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

def _dump_obj(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    try:
        return dict(obj)
    except Exception:
        return {"repr": repr(obj)}


def run_openai_style_finetune(args):
    from openai import OpenAI

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key environment variable: {args.api_key_env}")

    client_kwargs = {"api_key": api_key}
    if args.provider == "openai_compatible":
        if not args.api_base_url:
            raise ValueError("--api-base-url is required for --provider openai_compatible")
        client_kwargs["base_url"] = args.api_base_url

    client = OpenAI(**client_kwargs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_rows(args)
    train_df, val_df = split_train_val(df, args.val_ratio, args.seed)

    train_jsonl = out_dir / f"{args.task}_train.jsonl"
    n_train = write_chat_jsonl(train_df, args.task, str(train_jsonl))
    print(f"Wrote train JSONL: {train_jsonl} ({n_train} examples)")

    if n_train == 0:
        raise RuntimeError("No training examples were written.")

    val_jsonl = None
    n_val = 0
    if val_df is not None:
        val_jsonl = out_dir / f"{args.task}_val.jsonl"
        n_val = write_chat_jsonl(val_df, args.task, str(val_jsonl))
        print(f"Wrote val JSONL: {val_jsonl} ({n_val} examples)")

    print("Uploading training file...")
    with open(train_jsonl, "rb") as f:
        train_file = client.files.create(file=f, purpose="fine-tune")

    val_file = None
    if val_jsonl is not None and n_val > 0:
        print("Uploading validation file...")
        with open(val_jsonl, "rb") as f:
            val_file = client.files.create(file=f, purpose="fine-tune")

    job_kwargs = {
        "model": args.base_model,
        "training_file": train_file.id,
    }

    if val_file is not None:
        job_kwargs["validation_file"] = val_file.id

    if args.n_epochs is not None:
        job_kwargs["hyperparameters"] = {"n_epochs": args.n_epochs}

    if args.suffix:
        job_kwargs["suffix"] = args.suffix

    print("Creating fine-tuning job...")
    job = client.fine_tuning.jobs.create(**job_kwargs)

    job_path = out_dir / "api_finetune_job_initial.json"
    job_path.write_text(json.dumps(_dump_obj(job), indent=2), encoding="utf-8")

    print(f"Started fine-tuning job: {job.id}")
    print(f"Initial job metadata saved to: {job_path}")

    if args.no_wait:
        print("Not waiting for completion because --no-wait was set.")
        return

    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        status = getattr(job, "status", None)
        fine_tuned_model = getattr(job, "fine_tuned_model", None)

        print(f"status={status}, fine_tuned_model={fine_tuned_model}")

        if status in {"succeeded", "failed", "cancelled"}:
            break

        time.sleep(args.poll_seconds)

    final_job_path = out_dir / "api_finetune_job_final.json"
    final_job_path.write_text(json.dumps(_dump_obj(job), indent=2), encoding="utf-8")

    if getattr(job, "status", None) != "succeeded":
        raise RuntimeError(f"Fine-tuning did not succeed. Final status: {getattr(job, 'status', None)}")

    fine_tuned_model = getattr(job, "fine_tuned_model", None)
    if not fine_tuned_model:
        raise RuntimeError("Job succeeded but no fine_tuned_model id was returned.")

    model_path = out_dir / "fine_tuned_model.txt"
    model_path.write_text(fine_tuned_model + "\n", encoding="utf-8")

    config = {
        "provider": args.provider,
        "api_base_url": args.api_base_url,
        "base_model": args.base_model,
        "fine_tuned_model": fine_tuned_model,
        "task": args.task,
        "train_csv": args.train_csv,
        "n_train": n_train,
        "n_val": n_val,
        "seed": args.seed,
    }
    (out_dir / "frontier_finetune_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"Fine-tuned model id saved to: {model_path}")
    print(f"Fine-tuned model: {fine_tuned_model}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune frontier/API models for math task comparison."
    )

    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "openai_compatible"],
        help="Provider for fine-tuning. openai_compatible only works if the provider implements OpenAI-style fine-tuning endpoints.",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="Base URL for OpenAI-compatible provider, e.g. https://api.provider.com/v1",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model to fine-tune, e.g. gpt-4.1-mini or provider-specific model id.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["correct_answer", "next_subquestion", "distractor"],
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--suffix", default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--no-wait", action="store_true")

    args = parser.parse_args()

    if args.suffix is None:
        safe_task = args.task.replace("_", "-")
        args.suffix = f"{safe_task}"

    run_openai_style_finetune(args)


if __name__ == "__main__":
    main()