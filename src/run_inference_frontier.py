#!/usr/bin/env python3
"""
Run before/after inference for frontier/API models.

This script is intentionally separate from run_inference.py.

It supports:
  - OpenAI chat/completions-compatible generation
  - OpenAI-compatible generation endpoints
  - Anthropic generation

For after-mode:
  pass --checkpoint path/to/frontier_ft_dir
where the directory contains fine_tuned_model.txt created by finetune_frontier.py.

Example before:

python run_inference_frontier.py \
  --provider openai \
  --api-model gpt-4.1-mini \
  --model-name gpt-4.1-mini \
  --task correct_answer \
  --mode before \
  --data-csv out/correct_answer_pairs_gsm8k.csv \
  --out-dir out/frontier_eval

Example after:

python run_inference_frontier.py \
  --provider openai \
  --model-name gpt-4.1-mini \
  --task correct_answer \
  --mode after \
  --checkpoint out/frontier_ft/gpt41mini_correct_answer \
  --data-csv out/correct_answer_pairs_gsm8k.csv \
  --out-dir out/frontier_eval

Scoring:
  By default, this script uses an API judge for semantic yes/no judging.
  Use the SAME judge model across all frontier and local results if you want
  directly comparable scores.
"""

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Task constants and prompt builders
# ──────────────────────────────────────────────────────────────────────────────

TASK_TARGET_COL = {
    "correct_answer": "target_answer",
    "next_subquestion": "next_subquestion",
    "distractor": "target_distractor_answers",
}

TASK_MAX_NEW_TOKENS = {
    "correct_answer": 512,
    "next_subquestion": 80,
    "distractor": 512,
}

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


# ──────────────────────────────────────────────────────────────────────────────
# API generation
# ──────────────────────────────────────────────────────────────────────────────

def _openai_client(api_key_env: str, base_url: str | None = None):
    from openai import OpenAI

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key environment variable: {api_key_env}")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def _call_openai_chat(
    client,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    sleep_seconds: float,
    max_retries: int,
):
    last_err = None

    for attempt in range(max_retries):
        try:
            # Newer models may prefer max_completion_tokens.
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
            except TypeError:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            text = response.choices[0].message.content
            return "" if text is None else text.strip()

        except Exception as e:
            last_err = e
            wait = sleep_seconds * (2 ** attempt)
            print(f"[WARN] API call failed attempt {attempt + 1}/{max_retries}: {e}")
            print(f"       sleeping {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"API call failed after {max_retries} attempts: {last_err}")


def generate_openai_compatible(
    prompts,
    model_name: str,
    api_key_env: str,
    base_url: str | None,
    max_new_tokens: int,
    temperature: float,
    sleep_seconds: float,
    max_retries: int,
):
    client = _openai_client(api_key_env=api_key_env, base_url=base_url)

    outputs = []
    for prompt in tqdm(prompts, desc=f"Generating with {model_name}"):
        out = _call_openai_chat(
            client=client,
            model_name=model_name,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
        )
        outputs.append(out)

    return outputs


def generate_anthropic(
    prompts,
    model_name: str,
    api_key_env: str,
    max_new_tokens: int,
    temperature: float,
    sleep_seconds: float,
    max_retries: int,
):
    import anthropic

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key environment variable: {api_key_env}")

    client = anthropic.Anthropic(api_key=api_key)

    outputs = []
    for prompt in tqdm(prompts, desc=f"Generating with {model_name}"):
        last_err = None
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model_name,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )

                parts = []
                for block in response.content:
                    if getattr(block, "type", None) == "text":
                        parts.append(block.text)
                outputs.append("".join(parts).strip())
                break

            except Exception as e:
                last_err = e
                wait = sleep_seconds * (2 ** attempt)
                print(f"[WARN] Anthropic call failed attempt {attempt + 1}/{max_retries}: {e}")
                print(f"       sleeping {wait:.1f}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Anthropic call failed after {max_retries} attempts: {last_err}")

    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# Judge calls
# ──────────────────────────────────────────────────────────────────────────────

def judge_yes_no_api(
    judge_prompts,
    judge_provider: str,
    judge_model: str,
    judge_api_key_env: str,
    judge_api_base_url: str | None,
    temperature: float,
    sleep_seconds: float,
    max_retries: int,
):
    if judge_provider in {"openai", "openai_compatible"}:
        client = _openai_client(
            api_key_env=judge_api_key_env,
            base_url=judge_api_base_url if judge_provider == "openai_compatible" else None,
        )

        scores = []
        for prompt in tqdm(judge_prompts, desc=f"Judging with {judge_model}"):
            text = _call_openai_chat(
                client=client,
                model_name=judge_model,
                prompt=prompt,
                max_tokens=16,
                temperature=temperature,
                sleep_seconds=sleep_seconds,
                max_retries=max_retries,
            )
            verdicts = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
            final = verdicts[-1].lower() if verdicts else "no"
            scores.append(1 if final == "yes" else 0)
        return scores

    if judge_provider == "anthropic":
        import anthropic

        api_key = os.environ.get(judge_api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {judge_api_key_env}")

        client = anthropic.Anthropic(api_key=api_key)
        scores = []

        for prompt in tqdm(judge_prompts, desc=f"Judging with {judge_model}"):
            last_err = None
            for attempt in range(max_retries):
                try:
                    response = client.messages.create(
                        model=judge_model,
                        max_tokens=16,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = "".join(
                        block.text for block in response.content
                        if getattr(block, "type", None) == "text"
                    )
                    verdicts = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
                    final = verdicts[-1].lower() if verdicts else "no"
                    scores.append(1 if final == "yes" else 0)
                    break
                except Exception as e:
                    last_err = e
                    wait = sleep_seconds * (2 ** attempt)
                    print(f"[WARN] judge call failed attempt {attempt + 1}/{max_retries}: {e}")
                    print(f"       sleeping {wait:.1f}s")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Judge call failed after {max_retries} attempts: {last_err}")

        return scores

    raise ValueError(f"Unknown judge provider: {judge_provider}")


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"^(answer:|correct answer:|incorrect student answer:|distractor\s*\d*\s*:)\s*", "", text)
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_numeric_like(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?((\d+\.?\d*)|(\.\d+))", str(text).strip()))


def parse_distractor_gold(gold_str: str):
    try:
        parsed = ast.literal_eval(str(gold_str))
    except Exception:
        try:
            parsed = json.loads(str(gold_str))
        except Exception:
            parsed = [gold_str]

    return parsed if isinstance(parsed, list) else [parsed]


def score_correct_answer(
    predictions,
    golds,
    judge_provider,
    judge_model,
    judge_api_key_env,
    judge_api_base_url,
    judge_temperature,
    sleep_seconds,
    max_retries,
):
    scores = [None] * len(predictions)
    judge_prompts = []
    judge_indices = []

    for i, (pred, gold) in enumerate(zip(predictions, golds)):
        if is_numeric_like(gold):
            judge_prompts.append(
                "A student solved the following math problem and wrote this solution:\n"
                f"{pred}\n\n"
                f"The correct final answer is: {gold}\n\n"
                "Did the student arrive at the correct final answer? "
                "Answer only 'yes' or 'no'."
            )
            judge_indices.append(i)
        else:
            scores[i] = 1 if normalize_text(pred) == normalize_text(gold) else 0

    if judge_prompts:
        judge_scores = judge_yes_no_api(
            judge_prompts=judge_prompts,
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_api_key_env=judge_api_key_env,
            judge_api_base_url=judge_api_base_url,
            temperature=judge_temperature,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
        )
        for idx, score in zip(judge_indices, judge_scores):
            scores[idx] = score

    return scores


def score_next_subquestion(
    predictions,
    golds,
    judge_provider,
    judge_model,
    judge_api_key_env,
    judge_api_base_url,
    judge_temperature,
    sleep_seconds,
    max_retries,
):
    judge_prompts = [
        (
            "Are the following two math subquestions semantically equivalent?\n"
            "(They ask for exactly the same quantity, even if worded differently.)\n"
            "Answer only 'yes' or 'no'.\n\n"
            f"Question 1: {gold}\n"
            f"Question 2: {pred}"
        )
        for pred, gold in zip(predictions, golds)
    ]

    return judge_yes_no_api(
        judge_prompts=judge_prompts,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_api_key_env=judge_api_key_env,
        judge_api_base_url=judge_api_base_url,
        temperature=judge_temperature,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
    )


def build_answer_equivalence_prompt(problem_context: str, pred: str, gold: str) -> str:
    return (
        "You are judging whether two answers to a math problem are semantically equivalent.\n"
        "They do not need to use exactly the same wording, but they must express the same answer.\n"
        "Answer only 'yes' or 'no'.\n\n"
        f"Problem context:\n{problem_context}\n\n"
        f"Answer 1:\n{pred}\n\n"
        f"Answer 2:\n{gold}\n"
    )


def score_distractor(
    predictions,
    golds,
    prompts,
    prompt_styles,
    judge_provider,
    judge_model,
    judge_api_key_env,
    judge_api_base_url,
    judge_temperature,
    sleep_seconds,
    max_retries,
):
    scores = [None] * len(predictions)
    semantic_prompts = []
    semantic_slices = []

    for i, (pred, gold_str, prompt, prompt_style) in enumerate(
        zip(predictions, golds, prompts, prompt_styles)
    ):
        gold_values = parse_distractor_gold(gold_str)

        # Numeric MathGAP-style setting: match the final integer against gold distractor answers.
        if str(prompt_style) != "eedi_distractor":
            try:
                gold_nums = {int(float(value)) for value in gold_values}
                all_nums = re.findall(r"-?\b\d+\b", pred)
                if all_nums:
                    scores[i] = 1 if int(all_nums[-1]) in gold_nums else 0
                else:
                    scores[i] = 0
            except Exception:
                scores[i] = 0
            continue

        # Semantic EEDI-style setting.
        start = len(semantic_prompts)
        pred_norm = normalize_text(pred)
        for gold in gold_values:
            semantic_prompts.append(
                build_answer_equivalence_prompt(
                    problem_context=prompt,
                    pred=pred_norm,
                    gold=normalize_text(gold),
                )
            )
        end = len(semantic_prompts)
        semantic_slices.append((i, start, end))

    if semantic_prompts:
        semantic_scores = judge_yes_no_api(
            judge_prompts=semantic_prompts,
            judge_provider=judge_provider,
            judge_model=judge_model,
            judge_api_key_env=judge_api_key_env,
            judge_api_base_url=judge_api_base_url,
            temperature=judge_temperature,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
        )

        for row_idx, start, end in semantic_slices:
            scores[row_idx] = 1 if any(semantic_scores[start:end]) else 0

    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def resolve_after_model(checkpoint: str) -> str:
    """
    checkpoint can be either:
      1. a directory containing fine_tuned_model.txt
      2. a direct fine-tuned model id
    """
    if checkpoint is None:
        raise ValueError("--checkpoint is required for --mode after")

    path = Path(checkpoint)
    if path.is_dir():
        model_file = path / "fine_tuned_model.txt"
        if not model_file.exists():
            raise FileNotFoundError(
                f"{path} is a directory but does not contain fine_tuned_model.txt"
            )
        return model_file.read_text(encoding="utf-8").strip()

    return checkpoint


def sanitize_model_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def load_test_df(data_csv: str, max_rows: int | None):
    df = pd.read_csv(data_csv)

    if "split" in df.columns:
        df = df[df["split"] == "test"].copy().reset_index(drop=True)

    if max_rows is not None:
        df = df.head(max_rows).reset_index(drop=True)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Frontier/API inference and scoring for before/after eval."
    )

    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "openai_compatible", "anthropic"],
    )
    parser.add_argument(
        "--api-model",
        default=None,
        help="Base API model for before-mode. For after-mode, use --checkpoint.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Display/name prefix for output CSV, e.g. gpt-4.1-mini.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="Required for --provider openai_compatible.",
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=["correct_answer", "next_subquestion", "distractor"],
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["before", "after"],
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="For after-mode: directory containing fine_tuned_model.txt, or direct fine-tuned model id.",
    )
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--out-dir", default="out/frontier_eval")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--max-rows", type=int, default=None)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)  # kept for CLI compatibility
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=5)

    parser.add_argument(
        "--judge-provider",
        default="openai",
        choices=["openai", "openai_compatible", "anthropic"],
        help="Use a fixed judge across models for comparability.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4.1-mini",
        help="Judge model for yes/no scoring.",
    )
    parser.add_argument(
        "--judge-api-key-env",
        default="OPENAI_API_KEY",
    )
    parser.add_argument(
        "--judge-api-base-url",
        default=None,
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    if args.provider == "openai_compatible" and not args.api_base_url:
        raise ValueError("--api-base-url is required for --provider openai_compatible")

    if args.mode == "before":
        if not args.api_model:
            raise ValueError("--api-model is required for --mode before")
        generation_model = args.api_model
    else:
        generation_model = resolve_after_model(args.checkpoint)

    target_col = TASK_TARGET_COL[args.task]
    max_new_tokens = TASK_MAX_NEW_TOKENS[args.task]

    test_df = load_test_df(args.data_csv, args.max_rows)
    print(f"Task: {args.task} | Mode: {args.mode} | Test rows: {len(test_df)}")
    print(f"Generation model: {generation_model}")

    if target_col not in test_df.columns:
        raise ValueError(f"Missing target column for task {args.task}: {target_col}")

    valid_mask = test_df[target_col].notna()
    if "prompt" in test_df.columns:
        # Some EEDI rows may use question instead of prompt, so only drop prompt-NaNs
        # when neither prompt nor question exists.
        if "question" in test_df.columns:
            valid_mask = valid_mask & (test_df["prompt"].notna() | test_df["question"].notna())
        else:
            valid_mask = valid_mask & test_df["prompt"].notna()

    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        print(f"[WARN] dropping {n_dropped} rows with missing prompt/target")

    test_df = test_df[valid_mask].reset_index(drop=True)

    prompts = []
    kept_indices = []
    for idx, row in test_df.iterrows():
        try:
            prompts.append(build_prompt(row, args.task))
            kept_indices.append(idx)
        except Exception as e:
            print(f"[WARN] skipping row {idx}: {e}")

    test_df = test_df.iloc[kept_indices].reset_index(drop=True)
    golds = test_df[target_col].astype(str).tolist()

    if "prompt_style" in test_df.columns:
        prompt_styles = test_df["prompt_style"].astype(str).tolist()
    else:
        prompt_styles = [""] * len(test_df)

    # ── generation ─────────────────────────────────────────────────────────
    if args.provider in {"openai", "openai_compatible"}:
        predictions = generate_openai_compatible(
            prompts=prompts,
            model_name=generation_model,
            api_key_env=args.api_key_env,
            base_url=args.api_base_url if args.provider == "openai_compatible" else None,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )

    elif args.provider == "anthropic":
        predictions = generate_anthropic(
            prompts=prompts,
            model_name=generation_model,
            api_key_env=args.api_key_env,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )

    else:
        raise ValueError(f"Unknown provider: {args.provider}")

    # ── scoring ────────────────────────────────────────────────────────────
    if args.task == "correct_answer":
        scores = score_correct_answer(
            predictions=predictions,
            golds=golds,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            judge_api_key_env=args.judge_api_key_env,
            judge_api_base_url=args.judge_api_base_url,
            judge_temperature=args.judge_temperature,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )

    elif args.task == "next_subquestion":
        scores = score_next_subquestion(
            predictions=predictions,
            golds=golds,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            judge_api_key_env=args.judge_api_key_env,
            judge_api_base_url=args.judge_api_base_url,
            judge_temperature=args.judge_temperature,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )

    elif args.task == "distractor":
        scores = score_distractor(
            predictions=predictions,
            golds=golds,
            prompts=prompts,
            prompt_styles=prompt_styles,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            judge_api_key_env=args.judge_api_key_env,
            judge_api_base_url=args.judge_api_base_url,
            judge_temperature=args.judge_temperature,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )

    else:
        raise ValueError(f"Unknown task: {args.task}")

    # ── save results ───────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    model_short = sanitize_model_name(args.model_name)
    out_path = os.path.join(
        args.out_dir,
        f"{model_short}_{args.task}_{args.mode}{args.suffix}.csv",
    )

    id_cols = [c for c in ["example_idx", "pair_index", "id"] if c in test_df.columns]
    result_df = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)

    result_df["prediction"] = predictions
    result_df["score"] = scores

    accuracy = sum(scores) / len(scores) if scores else 0.0
    print(f"Accuracy ({args.mode}): {accuracy:.4f} ({sum(scores)}/{len(scores)})")

    result_df.to_csv(out_path, index=False)
    print(f"Saved results → {out_path}")

    summary = {
        "provider": args.provider,
        "generation_model": generation_model,
        "model_name": args.model_name,
        "task": args.task,
        "mode": args.mode,
        "data_csv": args.data_csv,
        "n": len(scores),
        "n_correct": int(sum(scores)),
        "accuracy": accuracy,
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "out_csv": out_path,
    }

    summary_path = out_path.replace(".csv", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary → {summary_path}")


if __name__ == "__main__":
    main()