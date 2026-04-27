#!/usr/bin/env python3
"""Run inference for one (model, task, mode) and save per-row results.

For task=correct_answer: chain-of-thought prompting; scored via Qwen 7B judge
  (checks whether the model's reasoning arrives at the correct numeric answer).
For task=next_subquestion: scores via Qwen 7B LLM judge (semantic equivalence).

Results are written to out/interim/<model_short>_<task>_<mode>.csv with columns:
    example_idx, pair_index, prediction, score

These interim files are later merged into the main task CSVs by merge_results.py.

Example (base model, before fine-tuning):
    python run_inference.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --task correct_answer \\
        --mode before

Example (fine-tuned, after):
    python run_inference.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --task correct_answer \\
        --mode after \\
        --checkpoint out/checkpoints/Qwen2.5-0.5B-Instruct_correct_answer/final
"""
import argparse
import json
import os
import re

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── constants ─────────────────────────────────────────────────────────────────

TASK_CSV = {
    "correct_answer": "out/correct_answer_pairs.csv",
    "next_subquestion": "out/next_subquestion_pairs.csv",
    "distractor": "out/distractor_pairs.csv",
}

TASK_TARGET_COL = {
    "correct_answer": "target_answer",
    "next_subquestion": "next_subquestion",
    "distractor": "target_distractor_answers",
}

TASK_MAX_NEW_TOKENS = {
    "correct_answer": 512,   # full chain-of-thought
    "next_subquestion": 80,
    "distractor": 512,       # erroneous chain-of-thought
}

LARGE_MODELS = {"7b", "8b"}

# Model used as judge for semantic equivalence of next_subquestion pairs
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Appended to every correct_answer prompt to elicit chain-of-thought
CORRECT_ANSWER_INSTRUCTION = (
    "\n\nSolve this problem step by step."
)

# Appended to every distractor prompt to elicit an erroneous chain-of-thought
DISTRACTOR_INSTRUCTION = (
    "\n\nIncorrect Answer: Let's think step by step."
)


def wrap_correct_answer_prompt(prompt: str) -> str:
    """Append the chain-of-thought instruction to a correct_answer prompt."""
    return prompt.rstrip() + CORRECT_ANSWER_INSTRUCTION


def wrap_distractor_prompt(prompt: str) -> str:
    """Append the erroneous-reasoning instruction to a distractor prompt."""
    return prompt.rstrip() + DISTRACTOR_INSTRUCTION


def _needs_4bit(model_name: str) -> bool:
    return any(tag in model_name.lower() for tag in LARGE_MODELS)


# ── scoring ───────────────────────────────────────────────────────────────────

def extract_number(text: str):
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(matches[0]) if matches else None


def score_correct_answer_local(
    predictions: list,
    golds: list,
    batch_size: int = 8,
) -> list:
    """Load Qwen 7B locally and judge whether each CoT prediction arrives at
    the correct numeric answer. Designed to be called after the inference
    model has been freed (del model + torch.cuda.empty_cache()).
    """
    judge_prompt_template = (
        "A student solved the following math problem and wrote this solution:\n"
        "{prediction}\n\n"
        "The correct final answer is: {gold}\n\n"
        "Did the student arrive at the correct final answer? "
        "Answer only 'yes' or 'no'."
    )

    print(f"Loading judge model: {JUDGE_MODEL_NAME}")
    judge_tokenizer = AutoTokenizer.from_pretrained(
        JUDGE_MODEL_NAME, trust_remote_code=True
    )
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token
    judge_tokenizer.padding_side = "left"

    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    judge_model.eval()

    scores = []
    for i in tqdm(range(0, len(predictions), batch_size), desc="LLM judge (correct_answer)"):
        batch_preds = predictions[i : i + batch_size]
        batch_golds = golds[i : i + batch_size]

        formatted = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": judge_prompt_template.format(
                    prediction=p, gold=g
                )}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p, g in zip(batch_preds, batch_golds)
        ]
        enc = judge_tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(judge_model.device)

        with torch.no_grad():
            out = judge_model.generate(
                **enc,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=judge_tokenizer.pad_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for seq in out:
            text = judge_tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip().lower()
            scores.append(1 if text.startswith("yes") else 0)

    del judge_model
    torch.cuda.empty_cache()
    return scores


def score_distractor_local(
    predictions: list,
    golds: list,
    batch_size: int = 8,
) -> list:
    """Load Qwen 7B locally and judge whether each distractor prediction arrives
    at ANY of the expected distractor (wrong) answers.  Each gold is a JSON
    string encoding a list of acceptable distractor answers.
    """
    judge_prompt_template = (
        "A student was asked to produce an incorrect answer (a distractor) "
        "for a math problem and wrote the following reasoning:\n"
        "{prediction}\n\n"
        "The expected distractor answers are: {gold}\n\n"
        "Did the student arrive at any of the expected distractor answers? "
        "Answer only 'yes' or 'no'."
    )

    print(f"Loading judge model: {JUDGE_MODEL_NAME}")
    judge_tokenizer = AutoTokenizer.from_pretrained(
        JUDGE_MODEL_NAME, trust_remote_code=True
    )
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token
    judge_tokenizer.padding_side = "left"

    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    judge_model.eval()

    scores = []
    for i in tqdm(range(0, len(predictions), batch_size), desc="LLM judge (distractor)"):
        batch_preds = predictions[i : i + batch_size]
        batch_golds = golds[i : i + batch_size]

        formatted = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": judge_prompt_template.format(
                    prediction=p, gold=g
                )}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p, g in zip(batch_preds, batch_golds)
        ]
        enc = judge_tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(judge_model.device)

        with torch.no_grad():
            out = judge_model.generate(
                **enc,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=judge_tokenizer.pad_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for seq in out:
            text = judge_tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip().lower()
            scores.append(1 if text.startswith("yes") else 0)

    del judge_model
    torch.cuda.empty_cache()
    return scores


def score_next_subquestion_local(
    predictions: list,
    golds: list,
    batch_size: int = 8,
) -> list:
    """Load Qwen 7B locally and judge semantic equivalence for all pairs.

    Designed to be called *after* the inference model has been freed
    (del model + torch.cuda.empty_cache()) so VRAM is available.
    """
    judge_prompt_template = (
        "Are the following two math subquestions semantically equivalent?\n"
        "(They ask for exactly the same quantity, even if worded differently.)\n"
        "Answer only 'yes' or 'no'.\n\n"
        "Question 1: {gold}\n"
        "Question 2: {pred}"
    )

    print(f"Loading judge model: {JUDGE_MODEL_NAME}")
    judge_tokenizer = AutoTokenizer.from_pretrained(
        JUDGE_MODEL_NAME, trust_remote_code=True
    )
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token
    judge_tokenizer.padding_side = "left"

    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    judge_model.eval()

    scores = []
    for i in tqdm(range(0, len(predictions), batch_size), desc="LLM judge"):
        batch_preds = predictions[i : i + batch_size]
        batch_golds = golds[i : i + batch_size]

        formatted = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": judge_prompt_template.format(
                    gold=g, pred=p
                )}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p, g in zip(batch_preds, batch_golds)
        ]
        enc = judge_tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(judge_model.device)

        with torch.no_grad():
            out = judge_model.generate(
                **enc,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=judge_tokenizer.pad_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for seq in out:
            text = judge_tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip().lower()
            scores.append(1 if text.startswith("yes") else 0)

    del judge_model
    torch.cuda.empty_cache()
    return scores


# ── generation ────────────────────────────────────────────────────────────────

def generate_predictions(
    model,
    tokenizer,
    prompts: list,
    max_new_tokens: int,
    batch_size: int = 8,
) -> list:
    all_outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
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
        description="Run inference for one (model, task, mode) and save to out/interim/."
    )
    parser.add_argument("--model-name", required=True,
                        help="HuggingFace model id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--task", required=True,
                        choices=["correct_answer", "next_subquestion", "distractor"])
    parser.add_argument("--mode", required=True, choices=["before", "after"],
                        help="'before' = base model, 'after' = fine-tuned")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to LoRA adapter directory (required when --mode after)")
    parser.add_argument("--data-csv", default=None,
                        help="Override default task CSV path")
    parser.add_argument("--out-dir", default="out/interim",
                        help="Directory to save interim result CSV")
    parser.add_argument("--suffix", default="",
                        help="Optional suffix appended to output filename, e.g. '_cot'")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Force 4-bit quantization (auto-enabled for 7B/8B)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit number of test rows (for smoke-testing)")
    args = parser.parse_args()

    if args.mode == "after" and not args.checkpoint:
        parser.error("--checkpoint is required when --mode=after")

    csv_path = args.data_csv or TASK_CSV[args.task]
    target_col = TASK_TARGET_COL[args.task]
    max_new_tokens = TASK_MAX_NEW_TOKENS[args.task]
    model_short = args.model_name.split("/")[-1]

    # ── load data (test rows only) ─────────────────────────────────────────
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError(
            "No 'split' column found. Run split_data.py first:\n"
            "  python split_data.py"
        )
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    if args.max_rows is not None:
        test_df = test_df.head(args.max_rows)
    print(f"Task: {args.task} | Mode: {args.mode} | Test rows: {len(test_df)}")

    # ── tokenizer ─────────────────────────────────────────────────────────
    # Always load from the original model name — LoRA fine-tuning does not
    # change the tokenizer, and checkpoint-saved tokenizer files may require
    # protobuf which is not always available.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
        model = model.merge_and_unload()

    model.eval()

    # ── generate ──────────────────────────────────────────────────────────
    # Drop rows where prompt or target is NaN (e.g. empty-tree rows not filtered upstream)
    valid_mask = test_df["prompt"].notna() & test_df[target_col].notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped:
        print(f"  WARNING: dropping {n_dropped} rows with NaN prompt/target")
    test_df = test_df[valid_mask].reset_index(drop=True)

    prompts = test_df["prompt"].astype(str).tolist()
    golds = test_df[target_col].astype(str).tolist()

    # For correct_answer, append a numeric-only instruction so the model
    # doesn't output sentences like "The answer is 12." — extract_number
    # then reliably grabs the lone digit.
    if args.task == "correct_answer":
        prompts = [wrap_correct_answer_prompt(p) for p in prompts]
    elif args.task == "distractor":
        prompts = [wrap_distractor_prompt(p) for p in prompts]

    predictions = generate_predictions(
        model, tokenizer, prompts, max_new_tokens, args.batch_size
    )

    # Free inference model before loading judge (saves VRAM for next_subquestion)
    del model
    torch.cuda.empty_cache()

    # ── score ─────────────────────────────────────────────────────────────
    if args.task == "correct_answer":
        scores = score_correct_answer_local(
            predictions, golds, batch_size=args.batch_size
        )
    elif args.task == "distractor":
        scores = score_distractor_local(
            predictions, golds, batch_size=args.batch_size
        )
    else:
        # next_subquestion: load Qwen 7B locally to judge semantic equivalence
        scores = score_next_subquestion_local(
            predictions, golds, batch_size=args.batch_size
        )

    # ── save interim results ──────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{model_short}_{args.task}_{args.mode}{args.suffix}.csv")

    id_cols = [c for c in ["example_idx", "pair_index"] if c in test_df.columns]
    result_df = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
    result_df["prediction"] = predictions
    result_df["score"] = scores
    result_df.to_csv(out_path, index=False)

    accuracy = sum(scores) / len(scores) if scores else 0.0
    print(f"Accuracy ({args.mode}): {accuracy:.4f}  ({sum(scores)}/{len(scores)})")
    print(f"Saved interim results → {out_path}")


if __name__ == "__main__":
    main()
