#!/usr/bin/env python3
"""Recompute best checkpoint for distractor task using regex scoring.

Loads each epoch checkpoint, generates val predictions, scores with regex
(last number in prediction vs gold distractor set), and updates
val_accuracy.json + best_checkpoint.json.

Usage:
    python src/recompute_distractor_best_ckpt.py \
        --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
        --train-csv colm-paper-code-cleaned/experiments/csm_mwps/out/distractor_pairs.csv \
        --checkpoint-dir out/distractor/results/checkpoints/Qwen2.5-0.5B-Instruct_distractor
"""
import argparse
import ast
import json
import os
import random
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Must match finetune.py exactly ────────────────────────────────────────────

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
        "Think step by step, flipping some comparison operations to arrive at "
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
        "Think step by step, flipping some transfer operations to arrive at "
        "a wrong answer."
    ),
}

TASK_MAX_NEW_TOKENS = {"distractor": 512}


def _build_prompt(row):
    mtype = row["misconception_type"]
    template = DISTRACTOR_PROMPT_BY_TYPE[mtype]
    return template.format(
        problem=row["problem"],
        correct_answer=row["correct_answer"],
    )


def build_val_data(df):
    prompts, golds = [], []
    for _, row in df.iterrows():
        prompts.append(_build_prompt(row))
        golds.append(row["target_distractor_answers"])
    return prompts, golds


def score_distractor_regex(predictions, golds):
    scores = []
    for pred, gold_str in zip(predictions, golds):
        gold_nums = set(ast.literal_eval(gold_str))
        all_nums = re.findall(r'-?\b\d+\b', pred)
        if all_nums:
            last_num = int(all_nums[-1])
            scores.append(1 if last_num in gold_nums else 0)
        else:
            scores.append(0)
    return scores


def generate_predictions(model, tokenizer, prompts, batch_size=4):
    max_new_tokens = TASK_MAX_NEW_TOKENS["distractor"]
    outputs = []
    model.eval()
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating val preds"):
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
            outputs.append(text.strip())
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Dir containing checkpoint-500, checkpoint-1000, etc.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    # Determine if 4-bit should be used (auto for 7b/8b)
    model_lower = args.model_name.lower()
    load_4bit = args.load_in_4bit or any(s in model_lower for s in ["7b", "8b"])

    quant_config = None
    if load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Build val set — must match finetune.py exactly (same split logic)
    df = pd.read_csv(args.train_csv)
    if "split" in df.columns:
        df = df[df["split"] == "train"].reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Val rows: {len(val_df)}")

    val_prompts, val_golds = build_val_data(val_df)

    # Discover checkpoints
    ckpt_dirs = sorted(Path(args.checkpoint_dir).glob("checkpoint-*"))
    if not ckpt_dirs:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return
    print(f"Found {len(ckpt_dirs)} checkpoints: {[d.name for d in ckpt_dirs]}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Generate predictions for each checkpoint
    all_predictions = {}
    for ckpt_dir in ckpt_dirs:
        print(f"\nGenerating predictions for {ckpt_dir.name}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if not load_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(ckpt_dir))
        model = model.merge_and_unload()
        model.eval()

        preds = generate_predictions(model, tokenizer, val_prompts, args.batch_size)
        all_predictions[str(ckpt_dir)] = preds

        del model
        torch.cuda.empty_cache()

    # Score with regex and find best
    best_acc, best_ckpt = -1.0, None
    for ckpt_path, preds in all_predictions.items():
        ckpt_name = Path(ckpt_path).name
        scores = score_distractor_regex(preds, val_golds)
        accuracy = sum(scores) / len(scores) if scores else 0.0
        print(f"  {ckpt_name}: regex accuracy = {accuracy:.4f} ({sum(scores)}/{len(scores)})")

        with open(os.path.join(ckpt_path, "val_accuracy.json"), "w") as f:
            json.dump({
                "checkpoint": ckpt_name,
                "accuracy": accuracy,
                "n_correct": sum(scores),
                "n_samples": len(scores),
            }, f, indent=2)

        if accuracy > best_acc:
            best_acc = accuracy
            best_ckpt = ckpt_path

    if best_ckpt:
        print(f"\nBest checkpoint: {best_ckpt} (accuracy={best_acc:.4f})")
        with open(os.path.join(args.checkpoint_dir, "best_checkpoint.json"), "w") as f:
            json.dump({"path": best_ckpt, "accuracy": best_acc}, f, indent=2)


if __name__ == "__main__":
    main()
