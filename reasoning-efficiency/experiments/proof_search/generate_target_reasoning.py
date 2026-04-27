#!/usr/bin/env python3
"""Add a 'target_question_reasoning_trace' column to correct_answer_pairs.csv.

Uses Qwen 7B with few-shot prompting to extract, from the full reasoning trace,
only the steps needed to answer the target question.

Each job processes a slice [--start-idx, --end-idx) and writes its results to a
chunk file in out/target_reasoning_chunks/. After all jobs finish, run with
--merge to combine chunks back into the main CSV.

Usage (single job):
    python generate_target_reasoning.py --start-idx 0 --end-idx 429

Usage (merge after all jobs done):
    python generate_target_reasoning.py --merge
"""
import argparse
import os
import glob

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TARGET_COL = "target_question_reasoning_trace"

# ── few-shot examples (built from real rows 0 and 2 of the CSV) ───────────────
# Each example shows: full_trace + target_question → trimmed trace

FEW_SHOT_EXAMPLES = [
    {
        "full_trace": (
            "Brant has 18 spinach leaves. Letti has 10 spinach leaves more than Harmon. "
            "The number of spinach leaves that Dario has more than Brant is the same as the difference between the number of spinach leaves that Letti has compared to Harmon. "
            "So Dario has 18 + 10 = 28 spinach leaves. "
            "Dario then gives Darn 16 spinach leaves. So Dario has 28 - 16 = 12 spinach leaves. "
            "Janice has 11 cucumbers. Janice has 12 cucumbers less than Quentin. So Quentin has 11 + 12 = 23 cucumbers. "
            "Tiphanie owns 19 cucumbers. Flossi has 6 cucumbers fewer than Tiphanie. So Flossi has 19 - 6 = 13 cucumbers. "
            "Christoforo possesses 14 cucumbers. So the difference between the number of cucumbers Tiphanie and Christoforo have is 19 - 14 = 5. "
            "The number of cucumbers that Queenie has more than Quentin is the same as the difference between the number of cucumbers that Tiphanie has compared to Christoforo. "
            "So Queenie has 23 + 5 = 28 cucumbers. "
            "If Dario and Queenie sum up the vegetables that they have, they have 12 + 28 = 40 in total."
        ),
        "target_question": "How many spinach leaves does Dario have in all?",
        "target_reasoning": (
            "Brant has 18 spinach leaves. Letti has 10 spinach leaves more than Harmon. "
            "The number of spinach leaves that Dario has more than Brant is the same as the difference between the number of spinach leaves that Letti has compared to Harmon. "
            "So Dario has 18 + 10 = 28 spinach leaves."
        ),
    },
    {
        "full_trace": (
            "Brant has 18 spinach leaves. Letti has 10 spinach leaves more than Harmon. "
            "The number of spinach leaves that Dario has more than Brant is the same as the difference between the number of spinach leaves that Letti has compared to Harmon. "
            "So Dario has 18 + 10 = 28 spinach leaves. "
            "Dario then gives Darn 16 spinach leaves. So Dario has 28 - 16 = 12 spinach leaves. "
            "Janice has 11 cucumbers. Janice has 12 cucumbers less than Quentin. So Quentin has 11 + 12 = 23 cucumbers. "
            "Tiphanie owns 19 cucumbers. Flossi has 6 cucumbers fewer than Tiphanie. So Flossi has 19 - 6 = 13 cucumbers. "
            "Christoforo possesses 14 cucumbers. So the difference between the number of cucumbers Tiphanie and Christoforo have is 19 - 14 = 5. "
            "The number of cucumbers that Queenie has more than Quentin is the same as the difference between the number of cucumbers that Tiphanie has compared to Christoforo. "
            "So Queenie has 23 + 5 = 28 cucumbers. "
            "If Dario and Queenie sum up the vegetables that they have, they have 12 + 28 = 40 in total."
        ),
        "target_question": "How many cucumbers does Tiphanie have more than Christoforo?",
        "target_reasoning": (
            "Tiphanie owns 19 cucumbers. Flossi has 6 cucumbers fewer than Tiphanie. So Flossi has 19 - 6 = 13 cucumbers. "
            "Christoforo possesses 14 cucumbers. So the difference between the number of cucumbers Tiphanie and Christoforo have is 19 - 14 = 5."
        ),
    },
]


def build_prompt(full_trace: str, target_question: str) -> str:
    lines = [
        "You are given a full reasoning trace that solves a multi-step math word problem, "
        "and a specific target question. Your task is to extract and output ONLY the reasoning "
        "steps from the full trace that are necessary to answer the target question. "
        "Do not add any new sentences. Do not include steps that are irrelevant to the target question. "
        "Output only the extracted reasoning steps, nothing else.\n"
    ]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"--- Example {i} ---")
        lines.append(f"Full reasoning trace:\n{ex['full_trace']}\n")
        lines.append(f"Target question: {ex['target_question']}\n")
        lines.append(f"Extracted reasoning:\n{ex['target_reasoning']}\n")

    lines.append("--- Your turn ---")
    lines.append(f"Full reasoning trace:\n{full_trace}\n")
    lines.append(f"Target question: {target_question}\n")
    lines.append("Extracted reasoning:")
    return "\n".join(lines)


def generate_batch(model, tokenizer, prompts: list, max_new_tokens: int = 512) -> list:
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
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
    results = []
    for seq in out:
        text = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
        results.append(text)
    return results


CHUNK_DIR = "out/target_reasoning_chunks"


def run_chunk(args) -> None:
    df = pd.read_csv(args.csv)
    total = len(df)

    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else total
    end = min(end, total)

    if start >= total:
        print(f"start-idx {start} >= total rows {total}, nothing to do.")
        return

    chunk_df = df.iloc[start:end].copy()
    print(f"Processing rows [{start}, {end}) — {len(chunk_df)} rows")

    os.makedirs(CHUNK_DIR, exist_ok=True)
    chunk_path = os.path.join(CHUNK_DIR, f"chunk_{start}_{end}.csv")

    # Resume: if chunk file exists, skip already-done rows
    if os.path.exists(chunk_path):
        done_df = pd.read_csv(chunk_path)
        done_orig_idx = set(done_df["orig_idx"].tolist())
        print(f"  Resuming: {len(done_orig_idx)} rows already in chunk file")
    else:
        done_orig_idx = set()

    todo = [
        (orig_idx, row)
        for orig_idx, row in zip(chunk_df.index.tolist(), chunk_df.itertuples())
        if orig_idx not in done_orig_idx
    ]
    print(f"  To process: {len(todo)} rows")

    if not todo:
        print("  All rows in this chunk already processed.")
        return

    # ── load model ────────────────────────────────────────────────────────
    print(f"Loading {JUDGE_MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME, trust_remote_code=True)
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

    # ── process in batches ────────────────────────────────────────────────
    results = []  # list of {orig_idx, TARGET_COL}
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(todo), batch_size), desc=f"chunk [{start},{end})"):
        batch = todo[batch_start : batch_start + batch_size]
        orig_indices = [orig_idx for orig_idx, _ in batch]
        prompts = [
            build_prompt(str(row.reasoning_trace), str(row.target_question))
            for _, row in batch
        ]

        outputs = generate_batch(model, tokenizer, prompts, max_new_tokens=args.max_new_tokens)

        for orig_idx, output in zip(orig_indices, outputs):
            results.append({"orig_idx": orig_idx, TARGET_COL: output})

        # Periodic save
        rows_done = batch_start + len(batch)
        if rows_done % args.save_every < batch_size or rows_done == len(todo):
            save_df = pd.DataFrame(results)
            if os.path.exists(chunk_path):
                existing = pd.read_csv(chunk_path)
                save_df = pd.concat([existing, save_df], ignore_index=True).drop_duplicates("orig_idx")
            save_df.to_csv(chunk_path, index=False)
            print(f"  [checkpoint] {rows_done}/{len(todo)} rows saved to {chunk_path}")

    print(f"Chunk done → {chunk_path}")


def run_merge(args) -> None:
    df = pd.read_csv(args.csv)
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = None

    chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.csv")))
    if not chunk_files:
        print(f"No chunk files found in {CHUNK_DIR}")
        return

    print(f"Merging {len(chunk_files)} chunk files ...")
    merged_rows = []
    for f in chunk_files:
        merged_rows.append(pd.read_csv(f))
    merged = pd.concat(merged_rows, ignore_index=True).drop_duplicates("orig_idx")
    merged = merged.set_index("orig_idx")

    filled = 0
    for orig_idx, row in merged.iterrows():
        if orig_idx in df.index:
            df.at[orig_idx, TARGET_COL] = row[TARGET_COL]
            filled += 1

    df.to_csv(args.csv, index=False)
    null_remaining = df[TARGET_COL].isna().sum()
    print(f"Merged {filled} rows into {args.csv}. Still null: {null_remaining}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="out/correct_answer_pairs.csv")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all chunk files back into the main CSV instead of generating")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="First row index (inclusive) to process")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Last row index (exclusive) to process")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save chunk file every N rows")
    args = parser.parse_args()

    if args.merge:
        run_merge(args)
    else:
        run_chunk(args)


if __name__ == "__main__":
    main()
