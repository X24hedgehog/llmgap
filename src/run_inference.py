#!/usr/bin/env python3
"""Unified inference + scoring for all tasks.

Auto-selects the best checkpoint via val_accuracy.json when --mode after.

Tasks:
  correct_answer   — CoT → judge checks numeric answer
  next_subquestion — predict next question → judge checks semantic equivalence
  distractor       — generate 1 distractor → judge checks if answer is in gold set

Example (base model):
    python run_inference.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --task correct_answer \\
        --mode before \\
        --data-csv path/to/correct_answer_pairs.csv

Example (fine-tuned, auto-select best checkpoint):
    python run_inference.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --task distractor \\
        --mode after \\
        --checkpoint out/checkpoints/Qwen2.5-0.5B-Instruct_distractor
"""
import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── constants ─────────────────────────────────────────────────────────────────

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

LARGE_MODELS = {"7b", "8b"}
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

CORRECT_ANSWER_INSTRUCTION = "\n\nSolve this problem step by step."

# Simplified distractor prompts (1 distractor, not full set)
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
        "the operation: they add instead of subtract (or vice versa), "
        "confusing the direction of the transfer.\n\n"
        "Your job is to simulate this mistake and produce a single distractor "
        "answer for the following math question.\n\n"
        "Question: {problem}\n"
        "Correct answer: {correct_answer}\n\n"
        "Think step by step, flipping one transfer operation to arrive at "
        "a wrong answer."
    ),
}


def _needs_4bit(model_name: str) -> bool:
    return any(tag in model_name.lower() for tag in LARGE_MODELS)


# ── prompt building ───────────────────────────────────────────────────────────

def _build_prompt(row: pd.Series, task: str) -> str:
    if task == "correct_answer":
        return str(row["prompt"]).rstrip() + CORRECT_ANSWER_INSTRUCTION
    elif task == "next_subquestion":
        return str(row["prompt"])
    elif task == "distractor":
        mtype = row.get("misconception_type", "")
        if mtype in DISTRACTOR_PROMPT_BY_TYPE:
            return DISTRACTOR_PROMPT_BY_TYPE[mtype].format(
                problem=row["problem"],
                correct_answer=row["correct_answer"],
            )
        # Fallback for CSVs without misconception_type
        return str(row["prompt"]).rstrip() + (
            "\n\nIncorrect Answer: Let's think step by step."
        )
    raise ValueError(f"Unknown task: {task}")


# ── checkpoint auto-selection ─────────────────────────────────────────────────

def find_best_checkpoint(checkpoint_dir: str) -> str:
    """Find the checkpoint with highest validation accuracy.

    Checks best_checkpoint.json first, then scans checkpoint-* dirs.
    Falls back to the directory itself (final model saved at top level).
    """
    best_file = os.path.join(checkpoint_dir, "best_checkpoint.json")
    if os.path.exists(best_file):
        with open(best_file) as f:
            info = json.load(f)
        path = info["path"]
        if os.path.isdir(path):
            print(f"Using best checkpoint from best_checkpoint.json: {path} "
                  f"(accuracy={info.get('accuracy', '?')})")
            return path

    best_acc, best_path = -1, None
    for d in sorted(Path(checkpoint_dir).glob("checkpoint-*")):
        vj = d / "val_accuracy.json"
        if vj.exists():
            info = json.loads(vj.read_text())
            if info["accuracy"] > best_acc:
                best_acc = info["accuracy"]
                best_path = str(d)

    if best_path:
        print(f"Auto-selected best checkpoint: {best_path} "
              f"(accuracy={best_acc:.4f})")
        return best_path

    # Fallback: use the directory itself (final adapter saved at top level)
    if os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
        print(f"Using final adapter at {checkpoint_dir}")
        return checkpoint_dir

    raise FileNotFoundError(
        f"No valid checkpoint found in {checkpoint_dir}. "
        "Expected best_checkpoint.json, checkpoint-*/val_accuracy.json, "
        "or adapter_config.json at top level."
    )


# ── scoring functions (Qwen 7B judge) ────────────────────────────────────────

def _load_judge():
    """Load the Qwen 7B judge model and tokenizer."""
    print(f"Loading judge model: {JUDGE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        JUDGE_MODEL_NAME, trust_remote_code=True
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


def _judge_yes_no(judge_model, judge_tokenizer, prompts, batch_size=8):
    """Run yes/no judge prompts and return list of 0/1 scores."""
    scores = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="LLM judge"):
        batch = prompts[i : i + batch_size]
        formatted = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
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
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=judge_tokenizer.pad_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for seq in out:
            text = judge_tokenizer.decode(
                seq[input_len:], skip_special_tokens=True
            ).strip().lower()
            scores.append(1 if text.startswith("yes") else 0)
    return scores


def score_correct_answer(predictions, golds, batch_size=8):
    """Judge: did the student arrive at the correct numeric answer?"""
    template = (
        "A student solved the following math problem and wrote this solution:\n"
        "{prediction}\n\n"
        "The correct final answer is: {gold}\n\n"
        "Did the student arrive at the correct final answer? "
        "Answer only 'yes' or 'no'."
    )
    judge_model, judge_tok = _load_judge()
    prompts = [
        template.format(prediction=p, gold=g)
        for p, g in zip(predictions, golds)
    ]
    scores = _judge_yes_no(judge_model, judge_tok, prompts, batch_size)
    del judge_model
    torch.cuda.empty_cache()
    return scores


def score_next_subquestion(predictions, golds, batch_size=8):
    """Judge: are the two subquestions semantically equivalent?"""
    template = (
        "Are the following two math subquestions semantically equivalent?\n"
        "(They ask for exactly the same quantity, even if worded differently.)\n"
        "Answer only 'yes' or 'no'.\n\n"
        "Question 1: {gold}\n"
        "Question 2: {pred}"
    )
    judge_model, judge_tok = _load_judge()
    prompts = [
        template.format(gold=g, pred=p)
        for p, g in zip(predictions, golds)
    ]
    scores = _judge_yes_no(judge_model, judge_tok, prompts, batch_size)
    del judge_model
    torch.cuda.empty_cache()
    return scores


def score_distractor(predictions, golds, batch_size=8):
    """Judge: did the student arrive at any of the expected distractor answers?

    Each gold is a JSON string encoding a list of acceptable distractor answers.
    Uses a 4-shot prompt (2 false-neg + 2 true-neg) to guide extraction.
    """
    template = (
        "You are a grading assistant. A student was asked to produce a distractor "
        "(an incorrect answer based on a specific misconception) for a math problem.\n\n"
        "Your task:\n"
        "1. Read the student's reasoning and identify the FINAL numeric answer they derived.\n"
        "2. Check if that number appears in the set of expected distractor answers.\n"
        "3. Answer 'yes' if it does, 'no' if it does not.\n\n"
        "Here are some examples:\n\n"
        "---\n"
        "Student reasoning:\n"
        "Liam has 52 pots. Then Liam gives Mia 17 pots. Applying the twist to this "
        "transfer operation. So Liam has 52 + 17 = 69 pots. Liam has 19 pots fewer "
        "than Jackson. So Jackson has 69 + 19 = 88 pots.\n"
        "Expected distractor answers: [88]\n"
        "The student's final answer is 88, which is in the expected set.\n"
        "Answer: yes\n\n"
        "---\n"
        "Student reasoning:\n"
        "Ava has 60 irons. Then Ava sells 8 irons to Charlotte. Applying the twist to "
        "this transfer operation. So Ava has 60 + 8 = 68 irons. Ava has 15 irons more "
        "than Liam. So Liam has 68 - 15 = 53 irons.\n"
        "Expected distractor answers: [53]\n"
        "The student's final answer is 53, which is in the expected set.\n"
        "Answer: yes\n\n"
        "---\n"
        "Student reasoning:\n"
        "Amelia has 52 toasters. Amelia has 21 toasters fewer than Christian. Applying "
        "the twist to this comparison operation. So Christian has 52 - 21 = 31 toasters. "
        "Then Christian gives Jackson 18 toasters. So Christian now has 31 - 18 = 13 "
        "toasters. Christian has 20 toasters more than Chloe. So Chloe has 13 - 20 = "
        "-7 toasters.\n"
        "Expected distractor answers: [75, 33]\n"
        "The student's final answer is -7, which is NOT in the expected set.\n"
        "Answer: no\n\n"
        "---\n"
        "Student reasoning:\n"
        "Avery has 57 green couches. Avery has 22 green couches more than Layla. So "
        "Layla has 57 - 22 = 35 green couches. Layla has 22 green couches fewer than "
        "Emily. Applying the twist to this comparison operation. So Emily has 35 - 22 = "
        "13 green couches. Emily has 22 green couches fewer than Mia. So Mia has "
        "13 - 22 = -9 green couches.\n"
        "Expected distractor answers: [123, 35, 79]\n"
        "The student's final answer is -9, which is NOT in the expected set.\n"
        "Answer: no\n\n"
        "---\n"
        "Now grade the following:\n\n"
        "Student reasoning:\n"
        "{prediction}\n\n"
        "Expected distractor answers: {gold}\n\n"
        "First identify the student's final numeric answer, then check if it is in "
        "the expected set. Answer only 'yes' or 'no'."
    )
    judge_model, judge_tok = _load_judge()
    prompts = [
        template.format(prediction=p, gold=g)
        for p, g in zip(predictions, golds)
    ]
    scores = _judge_yes_no(judge_model, judge_tok, prompts, batch_size)
    del judge_model
    torch.cuda.empty_cache()
    return scores


# ── generation ────────────────────────────────────────────────────────────────

def generate_predictions(model, tokenizer, prompts, max_new_tokens, batch_size=8):
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
            max_length=2048,
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
        description="Unified inference + scoring for all tasks."
    )
    parser.add_argument("--model-name", required=True,
                        help="HuggingFace model id")
    parser.add_argument("--task", required=True,
                        choices=["correct_answer", "next_subquestion", "distractor"])
    parser.add_argument("--mode", required=True, choices=["before", "after"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to LoRA adapter dir or parent dir with "
                             "checkpoint-* subdirs (auto-selects best)")
    parser.add_argument("--data-csv", required=True,
                        help="Path to task CSV")
    parser.add_argument("--out-dir", default="out/interim",
                        help="Directory to save result CSV")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit test rows (for smoke-testing)")
    args = parser.parse_args()

    if args.mode == "after" and not args.checkpoint:
        parser.error("--checkpoint is required when --mode=after")

    target_col = TASK_TARGET_COL[args.task]
    max_new_tokens = TASK_MAX_NEW_TOKENS[args.task]
    model_short = args.model_name.split("/")[-1]

    # ── resolve checkpoint ────────────────────────────────────────────────
    adapter_path = None
    if args.checkpoint:
        # If the dir has checkpoint-* subdirs, auto-select best
        has_subdirs = any(
            d.name.startswith("checkpoint-")
            for d in Path(args.checkpoint).iterdir()
            if d.is_dir()
        ) if Path(args.checkpoint).is_dir() else False

        if has_subdirs:
            adapter_path = find_best_checkpoint(args.checkpoint)
        else:
            adapter_path = args.checkpoint

    # ── load data (test rows) ─────────────────────────────────────────────
    df = pd.read_csv(args.data_csv)
    if "split" not in df.columns:
        raise ValueError("No 'split' column found in CSV.")
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    if args.max_rows is not None:
        test_df = test_df.head(args.max_rows)
    print(f"Task: {args.task} | Mode: {args.mode} | Test rows: {len(test_df)}")

    # Drop rows with NaN prompt/target
    valid_mask = test_df["prompt"].notna() & test_df[target_col].notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped:
        print(f"  WARNING: dropping {n_dropped} rows with NaN prompt/target")
    test_df = test_df[valid_mask].reset_index(drop=True)

    # ── tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
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

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()

    # ── build prompts ─────────────────────────────────────────────────────
    prompts = [_build_prompt(row, args.task) for _, row in test_df.iterrows()]
    golds = test_df[target_col].astype(str).tolist()

    predictions = generate_predictions(
        model, tokenizer, prompts, max_new_tokens, args.batch_size
    )

    del model
    torch.cuda.empty_cache()

    # ── score ─────────────────────────────────────────────────────────────
    if args.task == "correct_answer":
        scores = score_correct_answer(predictions, golds, args.batch_size)
    elif args.task == "next_subquestion":
        scores = score_next_subquestion(predictions, golds, args.batch_size)
    elif args.task == "distractor":
        scores = score_distractor(predictions, golds, args.batch_size)

    # ── save results ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"{model_short}_{args.task}_{args.mode}{args.suffix}.csv",
    )

    id_cols = [c for c in ["example_idx", "pair_index"] if c in test_df.columns]
    result_df = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
    result_df["prediction"] = predictions
    result_df["score"] = scores

    accuracy = sum(scores) / len(scores) if scores else 0.0
    print(f"Accuracy ({args.mode}): {accuracy:.4f}  ({sum(scores)}/{len(scores)})")

    result_df.to_csv(out_path, index=False)
    print(f"Saved results → {out_path}")


if __name__ == "__main__":
    main()
