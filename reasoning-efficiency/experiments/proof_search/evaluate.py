#!/usr/bin/env python3
"""Evaluate a model (base or fine-tuned) on a task's test split.

Scoring:
  correct_answer    — exact numeric match (parses first number from model output)
  next_subquestion  — normalised string match; optionally use OpenAI as judge
                      for cases that fail exact match

Example (before fine-tuning — base model):
    python evaluate.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --task correct_answer \\
        --out-path out/results/Qwen2.5-0.5B-Instruct_correct_answer_before.csv

Example (after fine-tuning):
    python evaluate.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --task correct_answer \\
        --checkpoint out/checkpoints/Qwen2.5-0.5B-Instruct_correct_answer/final \\
        --out-path out/results/Qwen2.5-0.5B-Instruct_correct_answer_after.csv
"""
import argparse
import json
import math
import os
import re

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── constants ─────────────────────────────────────────────────────────────────

TASK_TARGET_COL = {
    "correct_answer": "target_answer",
    "next_subquestion": "next_subquestion",
}

TASK_MAX_NEW_TOKENS = {
    "correct_answer": 16,
    "next_subquestion": 80,
}

LARGE_MODELS = {"7b", "8b"}


def _needs_4bit(model_name: str) -> bool:
    lower = model_name.lower()
    return any(tag in lower for tag in LARGE_MODELS)


# ── scoring helpers ───────────────────────────────────────────────────────────

def normalize_str(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def extract_number(text: str):
    """Return the first number found in text, or None."""
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(matches[0]) if matches else None


def score_correct_answer(prediction: str, gold) -> int:
    pred_num = extract_number(prediction)
    try:
        gold_num = float(gold)
    except (ValueError, TypeError):
        return 0
    if pred_num is None:
        return 0
    return int(math.isclose(pred_num, gold_num, rel_tol=1e-5))


def score_next_subquestion_exact(prediction: str, gold: str) -> int:
    return int(normalize_str(prediction) == normalize_str(gold))


def llm_judge_subquestions(
    predictions: list,
    golds: list,
    judge_model: str = "gpt-4o-mini",
    api_key: str = None,
) -> list:
    """
    Use OpenAI to judge semantic equivalence for pairs that failed exact match.
    Returns a list of 0/1 scores.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required for LLM judge: pip install openai")

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    scores = []
    for pred, gold in tqdm(zip(predictions, golds), total=len(predictions), desc="LLM judge"):
        prompt = (
            "Are the following two subquestions semantically equivalent "
            "(i.e., they ask the same mathematical quantity, even if worded differently)?\n"
            "Answer only 'yes' or 'no'.\n\n"
            f"Subquestion 1: {gold}\n"
            f"Subquestion 2: {pred}"
        )
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4,
            temperature=0,
        )
        answer = response.choices[0].message.content.strip().lower()
        scores.append(1 if answer.startswith("yes") else 0)
    return scores


# ── generation ────────────────────────────────────────────────────────────────

def generate_predictions(
    model,
    tokenizer,
    prompts: list,
    max_new_tokens: int,
    batch_size: int = 8,
) -> list:
    """Generate model completions for a list of user prompts in batches."""
    all_outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch_prompts
        ]
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for seq in out:
            text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            all_outputs.append(text.strip())

    return all_outputs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model (base or fine-tuned) on one task's test split."
    )
    parser.add_argument("--model-name", required=True,
                        help="HuggingFace model id (always the base model)")
    parser.add_argument("--task", required=True,
                        choices=["correct_answer", "next_subquestion"])
    parser.add_argument("--test-csv", default=None,
                        help="Path to test CSV (default: out/splits/<task>_test.csv)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned LoRA adapter directory. "
                             "Omit to evaluate the base model.")
    parser.add_argument("--out-path", required=True,
                        help="Output CSV path for per-row predictions and scores")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Force 4-bit quantization (auto-enabled for 7B+ models)")
    parser.add_argument("--use-llm-judge", action="store_true",
                        help="Use OpenAI to judge semantic equivalence for next_subquestion "
                             "(applied to rows that fail exact match)")
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Truncate test set (useful for quick smoke-tests)")
    args = parser.parse_args()

    if args.test_csv is None:
        args.test_csv = f"out/splits/{args.task}_test.csv"

    target_col = TASK_TARGET_COL[args.task]
    max_new_tokens = TASK_MAX_NEW_TOKENS[args.task]

    df = pd.read_csv(args.test_csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows)
    print(f"Evaluating {len(df)} rows from {args.test_csv}")

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ── tokenizer ─────────────────────────────────────────────────────────
    tok_source = args.checkpoint if args.checkpoint else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched left-pad during generation

    # ── model ─────────────────────────────────────────────────────────────
    load_4bit = args.load_in_4bit or _needs_4bit(args.model_name)
    quant_config = None
    if load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if not load_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.checkpoint:
        print(f"Loading LoRA adapter from {args.checkpoint}")
        model = PeftModel.from_pretrained(model, args.checkpoint)
        model = model.merge_and_unload()  # merge for faster inference

    model.eval()

    # ── generate ──────────────────────────────────────────────────────────
    prompts = df["prompt"].tolist()
    golds = df[target_col].astype(str).tolist()
    predictions = generate_predictions(
        model, tokenizer, prompts, max_new_tokens, args.batch_size
    )

    # ── score ─────────────────────────────────────────────────────────────
    if args.task == "correct_answer":
        scores = [
            score_correct_answer(pred, gold)
            for pred, gold in zip(predictions, golds)
        ]
    else:
        scores = [
            score_next_subquestion_exact(pred, gold)
            for pred, gold in zip(predictions, golds)
        ]
        if args.use_llm_judge:
            # Only call the judge on rows that failed exact match (saves cost)
            idx_to_judge = [i for i, s in enumerate(scores) if s == 0]
            if idx_to_judge:
                preds_to_judge = [predictions[i] for i in idx_to_judge]
                golds_to_judge = [golds[i] for i in idx_to_judge]
                llm_scores = llm_judge_subquestions(
                    preds_to_judge, golds_to_judge,
                    judge_model=args.judge_model,
                    api_key=args.openai_api_key,
                )
                for i, llm_score in zip(idx_to_judge, llm_scores):
                    scores[i] = max(scores[i], llm_score)

    # ── save results ──────────────────────────────────────────────────────
    results_df = df.copy()
    results_df["prediction"] = predictions
    results_df["score"] = scores
    results_df.to_csv(args.out_path, index=False)

    accuracy = sum(scores) / len(scores) if scores else 0.0
    print(f"Accuracy: {accuracy:.4f}  ({sum(scores)}/{len(scores)})")
    print(f"Saved predictions → {args.out_path}")

    summary = {
        "model_name": args.model_name,
        "task": args.task,
        "checkpoint": args.checkpoint,
        "mode": "after" if args.checkpoint else "before",
        "accuracy": accuracy,
        "n_correct": int(sum(scores)),
        "n_total": len(scores),
    }
    summary_path = args.out_path.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary    → {summary_path}")


if __name__ == "__main__":
    main()
