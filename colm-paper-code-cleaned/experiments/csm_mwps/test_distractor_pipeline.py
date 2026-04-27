#!/usr/bin/env python3
"""Test the distractor generation pipeline end-to-end on 10 examples.

For each of 3 Qwen models (0.5B, 1.5B, 3B Instruct), this script:
  1. Loads 10 test rows from distractor_pairs.csv
  2. Builds misconception-specific prompts
  3. Generates distractor sets
  4. Uses Qwen 7B judge to extract predicted number sets
  5. Computes TP, FP, FN, Jaccard per row
  6. Prints everything nicely for inspection

Usage:
    python test_distractor_pipeline.py
    python test_distractor_pipeline.py --n-rows 5
    python test_distractor_pipeline.py --data-csv out/distractor_pairs.csv
"""
import argparse
import json
import os
import textwrap

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

from run_inference import (
    JUDGE_MODEL_NAME,
    build_distractor_prompt,
    extract_number_set,
    generate_predictions,
)

W = 110  # print width


def hr(char="─"):
    print(char * W)


def section(title):
    hr("═")
    print(f"  {title}")
    hr("═")


def field(label, value, indent=2):
    prefix = " " * indent + f"{label}: "
    wrapped = textwrap.fill(
        str(value), width=W,
        initial_indent=prefix,
        subsequent_indent=" " * (indent + len(label) + 2),
    )
    print(wrapped)


# ── Judge: extract number sets ────────────────────────────────────────────────

def judge_extract_sets(predictions: list, batch_size: int = 4):
    """Load Qwen 7B and extract integer sets from raw predictions."""
    extraction_prompt = (
        "Extract all distinct integer answers from the student's response below.\n"
        "Return ONLY a JSON list of integers. If no numbers are found, return [].\n\n"
        "## Examples\n\n"
        "Student response: \"If we flip the comparison, Bob gets 30+12=42 and then 42+8=50, "
        "but flipping gives 30-12=18 so 18+8=26. Another flip gives 42-8=34.\"\n"
        "Output: [26, 34]\n\n"
        "Student response: \"[67, 34, 87]\"\n"
        "Output: [67, 34, 87]\n\n"
        "Student response: \"The distractor answers are 171 and 207.\"\n"
        "Output: [171, 207]\n\n"
        "Student response: \"I'm not sure about this problem.\"\n"
        "Output: []\n\n"
        "## Now extract from this response\n\n"
        "Student response: \"{prediction}\"\n"
        "Output:"
    )

    print(f"\nLoading judge model: {JUDGE_MODEL_NAME}")
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

    extracted = []
    judge_raw_outputs = []
    for i in tqdm(range(0, len(predictions), batch_size), desc="Judge extracting"):
        batch = predictions[i : i + batch_size]
        formatted = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": extraction_prompt.format(prediction=p)}],
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
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=judge_tokenizer.pad_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for seq in out:
            text = judge_tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            judge_raw_outputs.append(text.strip())
            extracted.append(extract_number_set(text))

    del judge_model
    torch.cuda.empty_cache()
    return extracted, judge_raw_outputs


def print_gold_set_stats(df: pd.DataFrame):
    """Print statistics on the length of gold distractor answer sets."""
    import numpy as np

    section("GOLD DISTRACTOR SET LENGTH STATISTICS")

    for split_name, split_df in [("all", df), ("train", df[df["split"] == "train"]), ("test", df[df["split"] == "test"])]:
        if len(split_df) == 0:
            continue
        lengths = split_df["target_distractor_answers"].apply(lambda x: len(json.loads(x)))

        hr()
        print(f"  Split: {split_name}  ({len(split_df)} rows)")
        print(f"    Mean:   {lengths.mean():.2f}")
        print(f"    Median: {lengths.median():.1f}")
        print(f"    Std:    {lengths.std():.2f}")
        print(f"    Min:    {lengths.min()}")
        print(f"    Max:    {lengths.max()}")
        print(f"    Distribution:")
        counts = lengths.value_counts().sort_index()
        for size, count in counts.items():
            pct = 100 * count / len(split_df)
            bar = "█" * int(pct / 2)
            print(f"      {size:3d} distractors: {count:5d} rows ({pct:5.1f}%) {bar}")

    # Per misconception type
    hr()
    print("  By misconception type:")
    for mtype in sorted(df["misconception_type"].unique()):
        sub = df[df["misconception_type"] == mtype]
        lengths = sub["target_distractor_answers"].apply(lambda x: len(json.loads(x)))
        print(f"    {mtype}")
        print(f"      Rows: {len(sub)}  Mean: {lengths.mean():.2f}  Median: {lengths.median():.1f}  Min: {lengths.min()}  Max: {lengths.max()}")

    hr("═")


def main():
    parser = argparse.ArgumentParser(description="Test distractor pipeline on a few examples.")
    parser.add_argument("--data-csv", default="out/distractor_pairs.csv")
    parser.add_argument("--n-rows", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling test rows")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only print gold distractor set length statistics, skip model inference")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_csv(args.data_csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if len(test_df) == 0:
        print("No test rows found!")
        return

    if args.stats_only:
        print_gold_set_stats(df)
        return

    # Sample n_rows, ensuring we get both misconception types if possible
    n = min(args.n_rows, len(test_df))
    sample_df = test_df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    print(f"Selected {len(sample_df)} test rows")
    print(f"  Misconception types: {sample_df['misconception_type'].value_counts().to_dict()}")

    # ── Build prompts ─────────────────────────────────────────────────────
    prompts = [build_distractor_prompt(row) for _, row in sample_df.iterrows()]

    # ── Models to test ────────────────────────────────────────────────────
    models_to_test = [
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "google/gemma-7b-it",
    ]

    # Store all predictions: {model_name: [pred_str, ...]}
    all_predictions = {}

    for model_name in models_to_test:
        print(f"\n{'='*W}")
        print(f"  Loading model: {model_name}")
        print(f"{'='*W}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        preds = generate_predictions(
            model, tokenizer, prompts,
            max_new_tokens=256,
            batch_size=args.batch_size,
        )
        all_predictions[model_name] = preds

        del model, tokenizer
        torch.cuda.empty_cache()

    # ── Judge: extract sets from all predictions ──────────────────────────
    # Flatten all predictions, run judge once, then split back
    flat_preds = []
    for model_name in models_to_test:
        flat_preds.extend(all_predictions[model_name])

    extracted_sets, judge_outputs = judge_extract_sets(flat_preds, batch_size=args.batch_size)

    # Split back by model
    extracted_by_model = {}
    judge_raw_by_model = {}
    offset = 0
    for model_name in models_to_test:
        count = len(all_predictions[model_name])
        extracted_by_model[model_name] = extracted_sets[offset : offset + count]
        judge_raw_by_model[model_name] = judge_outputs[offset : offset + count]
        offset += count

    # ── Pretty-print results ──────────────────────────────────────────────
    for row_idx in range(len(sample_df)):
        row = sample_df.iloc[row_idx]
        gold_set = {int(float(x)) for x in json.loads(row["target_distractor_answers"])}
        correct_ans = int(float(row["correct_answer"]))

        section(f"ROW {row_idx}  |  Misconception: {row['misconception_type']}")

        hr()
        print("  PROBLEM:")
        field("", row["problem"])

        hr()
        field("CORRECT ANSWER", correct_ans)

        hr()
        print("  CORRECT REASONING TRACE:")
        for line in str(row.get("reasoning_trace", "")).splitlines():
            print(f"    {line}")

        hr()
        field("GOLD DISTRACTOR SET", sorted(gold_set))
        field(f"  ({len(gold_set)} distractors)", "")

        hr()
        print("  PROMPT SENT TO LLM (first 300 chars):")
        prompt_preview = prompts[row_idx][:300]
        for line in prompt_preview.splitlines():
            print(f"    {line}")
        if len(prompts[row_idx]) > 300:
            print(f"    ... ({len(prompts[row_idx])} chars total)")

        # Per-model results
        for model_name in models_to_test:
            model_short = model_name.split("/")[-1]
            pred_raw = all_predictions[model_name][row_idx]
            pred_set = extracted_by_model[model_name][row_idx]
            judge_raw = judge_raw_by_model[model_name][row_idx]

            # Remove correct answer from predictions
            pred_set.discard(correct_ans)

            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            union = pred_set | gold_set
            jaccard = tp / len(union) if union else 1.0

            hr("·")
            print(f"  MODEL: {model_short}")
            print(f"    Raw output (first 500 chars):")
            for line in pred_raw[:500].splitlines():
                print(f"      {line}")
            if len(pred_raw) > 500:
                print(f"      ... ({len(pred_raw)} chars)")
            print(f"    Judge extracted: {judge_raw}")
            print(f"    Predicted set:   {sorted(pred_set)}")
            print(f"    Gold set:        {sorted(gold_set)}")
            print(f"    ────────────────────────────────")
            print(f"    TP={tp}  FP={fp}  FN={fn}  Jaccard={jaccard:.3f}")
            if pred_set & gold_set:
                print(f"    Correct distractors found:  {sorted(pred_set & gold_set)}")
            if pred_set - gold_set:
                print(f"    Spurious (FP):              {sorted(pred_set - gold_set)}")
            if gold_set - pred_set:
                print(f"    Missed (FN):                {sorted(gold_set - pred_set)}")

    # ── Aggregate summary ─────────────────────────────────────────────────
    section("AGGREGATE SUMMARY")
    for model_name in models_to_test:
        model_short = model_name.split("/")[-1]
        total_tp = total_fp = total_fn = 0
        jaccards = []

        for row_idx in range(len(sample_df)):
            row = sample_df.iloc[row_idx]
            gold_set = {int(float(x)) for x in json.loads(row["target_distractor_answers"])}
            correct_ans = int(float(row["correct_answer"]))
            pred_set = extracted_by_model[model_name][row_idx]
            pred_set.discard(correct_ans)

            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            union = pred_set | gold_set
            jaccard = tp / len(union) if union else 1.0

            total_tp += tp
            total_fp += fp
            total_fn += fn
            jaccards.append(jaccard)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0

        hr("·")
        print(f"  {model_short}")
        print(f"    Total TP={total_tp}  FP={total_fp}  FN={total_fn}")
        print(f"    Precision:    {precision:.4f}")
        print(f"    Recall:       {recall:.4f}")
        print(f"    Mean Jaccard: {mean_jaccard:.4f}")

    hr("═")


if __name__ == "__main__":
    main()
