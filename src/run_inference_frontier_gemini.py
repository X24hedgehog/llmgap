#!/usr/bin/env python3
"""
Gemini frontier inference + scoring for before-only evaluation.

This script is separate from run_inference.py and only uses API models.
It is designed for direct frontier-model evaluation, not fine-tuning.

Main fixes compared with the earlier version:
  - Uses Gemini 3.x thinking_level instead of thinking_budget=0
  - Falls back safely if thinking_config is rejected
  - Raises output-token budgets for math reasoning tasks
  - Logs Gemini usage metadata when available
  - Adds stricter task instructions to reduce verbose/truncated outputs
"""

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openai_judge import DEFAULT_OPENAI_EEDI_JUDGE_MODEL, judge_yes_no_openai
from prompt import build_answer_equivalence_prompt


TASK_TARGET_COL = {
    "correct_answer": "target_answer",
    "next_subquestion": "next_subquestion",
    "distractor": "target_distractor_answers",
}

TASK_MAX_NEW_TOKENS = {
    "correct_answer": 4096,
    "next_subquestion": 512,
    "distractor": 2048,
}

JUDGE_BATCH_SIZE = 8
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

CORRECT_ANSWER_INSTRUCTION = (
    "\n\nSolve the problem. Keep the reasoning concise. "
    "End with exactly one line of the form:\n"
    "#### <final numeric answer>"
)

NEXT_SUBQUESTION_INSTRUCTION = (
    "\n\nOutput only the next subquestion. "
    "Do not explain. Do not answer the subquestion."
)

DISTRACTOR_FINAL_INSTRUCTION = (
    "\n\nGenerate one plausible incorrect answer following the misconception. "
    "Keep the reasoning concise. End with exactly one line of the form:\n"
    "#### <incorrect numeric answer>"
)

DISTRACTOR_PROMPT_BY_TYPE = {
    "ContCompContMisconceptionIncons": (
        "You are an expert math educator generating a distractor based on "
        "the Comparison Inconsistency misconception.\n\n"
        "When a problem says \"A has N more than B\", the correct way to find B "
        "is B = A - N. A student with this misconception FLIPS the operation: "
        "they compute B = A + N instead. Similarly, \"fewer than\" means they subtract "
        "instead of add, \"times as many\" means they multiply instead of divide, etc.\n\n"
        "Your job is to simulate this mistake and produce a single distractor "
        "answer for the following math question.\n\n"
        "Question: {problem}\n"
        "Correct answer: {correct_answer}\n"
    ),
    "ContTransferContMisconceptionIncons": (
        "You are an expert math educator generating a distractor based on "
        "the Transfer Inconsistency misconception.\n\n"
        "When a problem says \"A gives B N items\", the correct effect on A "
        "is: A loses N, so subtract. A student with this misconception FLIPS "
        "the operation: they add instead of subtract, or vice versa, "
        "confusing the direction of the transfer.\n\n"
        "Your job is to simulate this mistake and produce a single distractor "
        "answer for the following math question.\n\n"
        "Question: {problem}\n"
        "Correct answer: {correct_answer}\n"
    ),
}


def safe_get(row: pd.Series, key: str, default: str = "") -> str:
    if key not in row or pd.isna(row[key]):
        return default
    return str(row[key])


def build_prompt(row: pd.Series, task: str) -> str:
    prompt_style = safe_get(row, "prompt_style")

    if prompt_style == "eedi_correct_answer":
        return (
            f"Question: {safe_get(row, 'question').strip()}"
            + CORRECT_ANSWER_INSTRUCTION
        )

    if prompt_style == "eedi_distractor":
        return (
            "You are solving a math question as a student with the following "
            f"misconception: {safe_get(row, 'misconception_name').strip()}\n\n"
            f"Question: {safe_get(row, 'question').strip()}\n\n"
            "Think step by step and give the incorrect answer this student "
            "would produce."
        )

    if task == "correct_answer":
        if "prompt" in row and not pd.isna(row["prompt"]):
            return str(row["prompt"]).rstrip() + CORRECT_ANSWER_INSTRUCTION
        if "question" in row and not pd.isna(row["question"]):
            return (
                f"Question: {str(row['question']).strip()}"
                + CORRECT_ANSWER_INSTRUCTION
            )
        raise ValueError("correct_answer row has neither prompt nor question")

    if task == "next_subquestion":
        if "prompt" not in row or pd.isna(row["prompt"]):
            raise ValueError("next_subquestion row requires prompt")
        return str(row["prompt"]).rstrip() + NEXT_SUBQUESTION_INSTRUCTION

    if task == "distractor":
        misconception_type = safe_get(row, "misconception_type")

        if misconception_type in DISTRACTOR_PROMPT_BY_TYPE:
            return (
                DISTRACTOR_PROMPT_BY_TYPE[misconception_type].format(
                    problem=safe_get(row, "problem"),
                    correct_answer=safe_get(row, "correct_answer"),
                )
                + DISTRACTOR_FINAL_INSTRUCTION
            )

        if "prompt" in row and not pd.isna(row["prompt"]):
            return str(row["prompt"]).rstrip() + DISTRACTOR_FINAL_INSTRUCTION

        if "question" in row and not pd.isna(row["question"]):
            return (
                f"Question: {str(row['question']).strip()}\n\n"
                "Generate one plausible incorrect answer. "
                "Keep the reasoning concise."
                + DISTRACTOR_FINAL_INSTRUCTION
            )

        raise ValueError("distractor row has no usable prompt")

    raise ValueError(f"Unknown task: {task}")


def get_gemini_client(api_key_env: str):
    from google import genai

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key environment variable: {api_key_env}")

    return genai.Client(api_key=api_key)


def usage_to_dict(usage: Any) -> dict:
    if usage is None:
        return {}

    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            pass

    keys = [
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "total_token_count",
        "cached_content_token_count",
    ]
    out = {}
    for key in keys:
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    return out


def extract_gemini_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""

    parts_out = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None)
        if not parts:
            continue
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                parts_out.append(str(part_text))

    return "".join(parts_out).strip()


def make_gemini_config(
    model_name: str,
    max_output_tokens: int,
    temperature: float,
    thinking_level: str,
    use_thinking_config: bool,
):
    from google.genai import types

    config_kwargs = {
        "max_output_tokens": max_output_tokens,
    }

    if temperature is not None:
        config_kwargs["temperature"] = temperature

    if use_thinking_config and thinking_level != "none":
        if model_name.startswith("gemini-3"):
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level
            )
        elif model_name.startswith("gemini-2.5-flash"):
            if thinking_level == "off":
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=0
                )
            else:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=-1
                )

    return types.GenerateContentConfig(**config_kwargs)


def call_gemini(
    client,
    model_name: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    sleep_seconds: float,
    max_retries: int,
    thinking_level: str,
) -> tuple[str, dict]:
    last_err = None
    use_thinking_config = thinking_level != "none"

    for attempt in range(max_retries):
        try:
            config = make_gemini_config(
                model_name=model_name,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                thinking_level=thinking_level,
                use_thinking_config=use_thinking_config,
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            text = extract_gemini_text(response)
            usage = usage_to_dict(getattr(response, "usage_metadata", None))
            return text, usage

        except Exception as e:
            last_err = e
            err = str(e)

            if use_thinking_config and (
                "INVALID_ARGUMENT" in err
                or "thinking" in err.lower()
                or "ThinkingConfig" in err
            ):
                print(
                    "[WARN] thinking_config rejected by API, retrying without it: "
                    f"{e}"
                )
                use_thinking_config = False
                continue

            wait = sleep_seconds * (2 ** attempt)
            print(f"[WARN] Gemini call failed attempt {attempt + 1}/{max_retries}: {e}")
            print(f"       sleeping {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


def generate_gemini_predictions(
    prompts,
    model_name: str,
    api_key_env: str,
    max_new_tokens: int,
    temperature: float,
    sleep_seconds: float,
    max_retries: int,
    thinking_level: str,
):
    client = get_gemini_client(api_key_env)

    outputs = []
    usages = []

    for prompt in tqdm(prompts, desc=f"Generating with {model_name}"):
        output, usage = call_gemini(
            client=client,
            model_name=model_name,
            prompt=prompt,
            max_output_tokens=max_new_tokens,
            temperature=temperature,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            thinking_level=thinking_level,
        )
        outputs.append(output)
        usages.append(usage)

    return outputs, usages


def _load_judge():
    print(f"Loading judge model: {JUDGE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        JUDGE_MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _judge_yes_no(judge_model, judge_tokenizer, prompts, batch_size=JUDGE_BATCH_SIZE):
    scores = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="LLM judge"):
        batch = prompts[i : i + batch_size]
        formatted = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch
        ]
        enc = judge_tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(judge_model.device)

        with torch.no_grad():
            out = judge_model.generate(
                **enc,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=judge_tokenizer.pad_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for seq in out:
            text = judge_tokenizer.decode(
                seq[input_len:],
                skip_special_tokens=True,
            ).strip()
            verdicts = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
            final_verdict = verdicts[-1].lower() if verdicts else "no"
            scores.append(1 if final_verdict == "yes" else 0)

    return scores


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(
        r"^(answer:|correct answer:|incorrect student answer:|distractor\s*\d*\s*:)\s*",
        "",
        text,
    )
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


def use_semantic_distractor_scoring(prompt_style: str) -> bool:
    return str(prompt_style) == "eedi_distractor"


def use_semantic_correct_answer_scoring(prompt_style: str) -> bool:
    return str(prompt_style) == "eedi_correct_answer"


def score_correct_answer(
    predictions,
    golds,
    prompts,
    prompt_styles,
    batch_size: int = JUDGE_BATCH_SIZE,
):
    scores = [None] * len(predictions)
    numeric_examples = []
    semantic_examples = []

    for i, (pred, gold, prompt, prompt_style) in enumerate(
        zip(predictions, golds, prompts, prompt_styles)
    ):
        if use_semantic_correct_answer_scoring(prompt_style):
            semantic_examples.append((i, prompt, pred, gold))
            continue

        if is_numeric_like(gold):
            numeric_examples.append((i, pred, gold))
            continue

        scores[i] = 1 if normalize_text(pred) == normalize_text(gold) else 0

    if numeric_examples:
        judge_prompts = [
            (
                "A student solved the following math problem and wrote this solution:\n"
                f"{pred}\n\n"
                f"The correct final answer is: {gold}\n\n"
                "Did the student arrive at the correct final answer? "
                "Answer only 'yes' or 'no'."
            )
            for _, pred, gold in numeric_examples
        ]
        judge_model, judge_tok = _load_judge()
        judge_scores = _judge_yes_no(
            judge_model,
            judge_tok,
            judge_prompts,
            batch_size,
        )
        del judge_model
        torch.cuda.empty_cache()

        for (idx, _, _), score in zip(numeric_examples, judge_scores):
            scores[idx] = score

    if semantic_examples:
        semantic_prompts = [
            build_answer_equivalence_prompt(
                prompt,
                normalize_text(pred),
                normalize_text(gold),
            )
            for _, prompt, pred, gold in semantic_examples
        ]
        print(f"Using OpenAI EEDI judge model: {DEFAULT_OPENAI_EEDI_JUDGE_MODEL}")
        semantic_scores = judge_yes_no_openai(
            semantic_prompts,
            batch_size=batch_size,
        )

        for (idx, _, _, _), score in zip(semantic_examples, semantic_scores):
            scores[idx] = score

    return scores


def score_next_subquestion(
    predictions,
    golds,
    batch_size: int = JUDGE_BATCH_SIZE,
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

    judge_model, judge_tok = _load_judge()
    scores = _judge_yes_no(judge_model, judge_tok, judge_prompts, batch_size)
    del judge_model
    torch.cuda.empty_cache()
    return scores


def score_distractor(
    predictions,
    golds,
    prompts,
    prompt_styles,
    batch_size: int = JUDGE_BATCH_SIZE,
):
    scores = [None] * len(predictions)
    semantic_examples = []

    for i, (pred, gold_str, prompt, prompt_style) in enumerate(
        zip(predictions, golds, prompts, prompt_styles)
    ):
        gold_values = parse_distractor_gold(gold_str)

        if not use_semantic_distractor_scoring(prompt_style):
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

        semantic_examples.append((i, prompt, pred, gold_values))

    if semantic_examples:
        semantic_prompts = []
        semantic_slices = []

        for row_idx, prompt, pred, gold_values in semantic_examples:
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
            semantic_slices.append((row_idx, start, len(semantic_prompts)))

        print(f"Using OpenAI EEDI judge model: {DEFAULT_OPENAI_EEDI_JUDGE_MODEL}")
        semantic_scores = judge_yes_no_openai(
            semantic_prompts,
            batch_size=batch_size,
        )

        for row_idx, start, end in semantic_slices:
            scores[row_idx] = 1 if any(semantic_scores[start:end]) else 0

    return scores

def print_smoke_examples(
    prompts,
    predictions,
    golds,
    scores,
    generation_usages,
    max_print: int = 10,
):
    print("\n" + "=" * 100)
    print("SMOKE TEST EXAMPLES")
    print("=" * 100)

    n = min(len(predictions), max_print)

    for i in range(n):
        print("\n" + "-" * 100)
        print(f"Example {i + 1}/{len(predictions)}")
        print("-" * 100)

        print("\n[PROMPT]")
        print(prompts[i][:2000])
        if len(prompts[i]) > 2000:
            print("\n... [prompt truncated in terminal] ...")

        print("\n[GOLD]")
        print(golds[i])

        print("\n[GENERATION]")
        print(predictions[i])

        print("\n[SCORE]")
        print(scores[i])

        if generation_usages:
            print("\n[GENERATION USAGE]")
            print(json.dumps(generation_usages[i], indent=2, ensure_ascii=False))

    print("\n" + "=" * 100)
    print("END SMOKE TEST EXAMPLES")
    print("=" * 100 + "\n")

def sanitize_model_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def load_test_df(data_csv: str, max_rows):
    df = pd.read_csv(data_csv)

    if "split" in df.columns:
        df = df[df["split"] == "test"].copy().reset_index(drop=True)

    if max_rows is not None:
        df = df.head(max_rows).reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Gemini frontier inference and scoring"
    )

    parser.add_argument("--api-model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")

    parser.add_argument(
        "--task",
        required=True,
        choices=["correct_answer", "next_subquestion", "distractor"],
    )
    parser.add_argument(
        "--mode",
        default="before",
        choices=["before"],
        help="Frontier models are evaluated directly only",
    )
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--out-dir", default="out/frontier_gemini")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--max-rows", type=int, default=None)

    parser.add_argument(
    "--print-examples",
    action="store_true",
    help="Print prompts, generations, golds, scores, and usage metadata to terminal",
    )
    parser.add_argument(
        "--max-print-examples",
        type=int,
        default=10,
        help="Maximum number of examples to print when --print-examples is enabled",
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=5)

    parser.add_argument(
        "--thinking-level",
        choices=["none", "off", "minimal", "low", "medium", "high"],
        default="minimal",
        help=(
            "For Gemini 3.x, use minimal/low/medium/high. "
            "For Gemini 2.5 Flash, off maps to thinking_budget=0. "
            "Use none to omit thinking_config entirely."
        ),
    )
    parser.add_argument(
        "--judge-thinking-level",
        choices=["none", "off", "minimal", "low", "medium", "high"],
        default="minimal",
    )

    parser.add_argument(
        "--judge-provider",
        choices=["gemini", "openai"],
        default="gemini",
        help="Deprecated and ignored. Frontier scoring now reuses run_inference.py judges.",
    )
    parser.add_argument(
        "--judge-model",
        default="gemini-3.6-flash",
        help="Deprecated and ignored. Frontier scoring now reuses run_inference.py judges.",
    )
    parser.add_argument(
        "--judge-gemini-api-key-env",
        default="GEMINI_API_KEY",
        help="Deprecated and ignored. Frontier scoring now reuses run_inference.py judges.",
    )
    parser.add_argument(
        "--judge-openai-api-key-env",
        default="OPENAI_API_KEY",
        help="Deprecated and ignored. Frontier scoring now reuses run_inference.py judges.",
    )

    args = parser.parse_args()

    target_col = TASK_TARGET_COL[args.task]
    max_new_tokens = TASK_MAX_NEW_TOKENS[args.task]

    test_df = load_test_df(args.data_csv, args.max_rows)
    print(
        f"Task: {args.task} | Mode: {args.mode} | "
        f"Test rows before filtering: {len(test_df)}"
    )
    print(f"Generation model: {args.api_model}")
    print(f"Thinking level: {args.thinking_level}")
    print(
        "Scoring judges: "
        f"{JUDGE_MODEL_NAME} for yes/no checks, "
        f"{DEFAULT_OPENAI_EEDI_JUDGE_MODEL} for EEDI semantic equivalence"
    )

    if target_col not in test_df.columns:
        raise ValueError(f"Missing target column for task {args.task}: {target_col}")

    valid_mask = test_df[target_col].notna()

    if "prompt" in test_df.columns:
        if "question" in test_df.columns:
            valid_mask = valid_mask & (
                test_df["prompt"].notna() | test_df["question"].notna()
            )
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

    print(f"Final test rows: {len(test_df)}")

    predictions, generation_usages = generate_gemini_predictions(
        prompts=prompts,
        model_name=args.api_model,
        api_key_env=args.api_key_env,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        thinking_level=args.thinking_level,
    )

    if args.task == "correct_answer":
        scores = score_correct_answer(
            predictions=predictions,
            golds=golds,
            prompts=prompts,
            prompt_styles=prompt_styles,
        )

    elif args.task == "next_subquestion":
        scores = score_next_subquestion(
            predictions=predictions,
            golds=golds,
        )

    elif args.task == "distractor":
        scores = score_distractor(
            predictions=predictions,
            golds=golds,
            prompts=prompts,
            prompt_styles=prompt_styles,
        )

    else:
        raise ValueError(f"Unknown task: {args.task}")

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
    result_df["generation_usage_metadata"] = [
        json.dumps(u, ensure_ascii=False) for u in generation_usages
    ]

    accuracy = sum(scores) / len(scores) if scores else 0.0
    print(f"Accuracy ({args.mode}): {accuracy:.4f} ({sum(scores)}/{len(scores)})")

    if args.print_examples:
        print_smoke_examples(
            prompts=prompts,
            predictions=predictions,
            golds=golds,
            scores=scores,
            generation_usages=generation_usages,
            max_print=args.max_print_examples,
        )

    result_df.to_csv(out_path, index=False)
    print(f"Saved results -> {out_path}")

    total_usage = {}
    for usage in generation_usages:
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                total_usage[key] = total_usage.get(key, 0) + value

    summary = {
        "provider": "gemini",
        "generation_model": args.api_model,
        "model_name": args.model_name,
        "task": args.task,
        "mode": args.mode,
        "data_csv": args.data_csv,
        "n": len(scores),
        "n_correct": int(sum(scores)),
        "accuracy": accuracy,
        "judge_model": JUDGE_MODEL_NAME,
        "eedi_semantic_judge_model": DEFAULT_OPENAI_EEDI_JUDGE_MODEL,
        "thinking_level": args.thinking_level,
        "total_generation_usage_metadata": total_usage,
        "out_csv": out_path,
    }

    summary_path = out_path.replace(".csv", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()