#!/usr/bin/env python3
"""Run inference for one (model, task, mode) and save per-row results.

Tasks supported:
  correct_answer — chain-of-thought prompting; scored via Qwen 7B judge
  distractor     — misconception-conditioned distractor generation;
                   scored via Qwen 7B judge comparing SETS of numbers
                   (Jaccard, TP, FP, FN)

Results are written to out/interim/<model_short>_<task>_<mode>.csv.

Example (base model, before fine-tuning):
    python run_inference.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --task correct_answer \\
        --mode before

Example (fine-tuned, after):
    python run_inference.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --task distractor \\
        --mode after \\
        --checkpoint out/checkpoints/Qwen2.5-0.5B-Instruct_distractor/final
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
    "correct_answer": "out/correct_answer_distractor_pairs.csv",
    "distractor": "out/distractor_pairs.csv",
}

TASK_TARGET_COL = {
    "correct_answer": "target_answer",
    "distractor": "target_distractor_answers",
}

TASK_MAX_NEW_TOKENS = {
    "correct_answer": 512,
    "distractor": 256,
}

LARGE_MODELS = {"7b", "8b"}

JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ── Prompt templates ──────────────────────────────────────────────────────────

from prompts import CORRECT_ANSWER_INSTRUCTION, DISTRACTOR_PROMPT_BY_TYPE


def wrap_correct_answer_prompt(prompt: str) -> str:
    return prompt.rstrip() + CORRECT_ANSWER_INSTRUCTION


def build_distractor_prompt(row: pd.Series) -> str:
    """Build a misconception-specific distractor prompt from the row data."""
    mtype = row["misconception_type"]
    template = DISTRACTOR_PROMPT_BY_TYPE[mtype]
    return template.format(
        problem=row["problem"],
        correct_answer=row["correct_answer"],
    )


def _needs_4bit(model_name: str) -> bool:
    return any(tag in model_name.lower() for tag in LARGE_MODELS)


# ── number extraction ─────────────────────────────────────────────────────────

def extract_number_set(text: str) -> set:
    """Extract a set of integers from LLM output.

    Tries JSON list first, then falls back to regex.
    """
    text = text.strip()
    # Try to parse the whole text as a JSON list
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return {int(float(x)) for x in parsed if x is not None}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Try to find a JSON-like list inside the text
    match = re.search(r'\[[\d\s,.\-]+\]', text)
    if match:
        try:
            parsed = json.loads(match.group())
            return {int(float(x)) for x in parsed}
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: extract all integers from the text
    nums = re.findall(r'-?\d+', text)
    return {int(x) for x in nums} if nums else set()


# ── scoring: correct answer ───────────────────────────────────────────────────

def score_correct_answer_local(
    predictions: list,
    golds: list,
    batch_size: int = 8,
) -> list:
    """Qwen 7B judge: did the student arrive at the correct numeric answer?"""
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


# ── scoring: distractor (set comparison) ──────────────────────────────────────

def score_distractor_set(
    predictions: list,
    golds: list,
    correct_answers: list,
    batch_size: int = 4,
) -> list:
    """Use Qwen 7B to extract the predicted set of distractor numbers, then
    compute set-level metrics (TP, FP, FN, Jaccard) against the gold set.

    The judge is given the model's raw text output and asked to extract the
    set of distinct integer answers the model produced. We use few-shot
    examples so Qwen 7B reliably returns a JSON list.

    Returns a list of dicts, one per row:
        {"pred_set": [...], "gold_set": [...], "tp": int, "fp": int,
         "fn": int, "jaccard": float}
    """
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

    print(f"Loading judge model for set extraction: {JUDGE_MODEL_NAME}")
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

    # Step 1: extract predicted sets via judge
    extracted_pred_sets = []
    for i in tqdm(range(0, len(predictions), batch_size), desc="Judge extracting sets"):
        batch_preds = predictions[i : i + batch_size]

        formatted = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": extraction_prompt.format(prediction=p)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch_preds
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
            extracted_pred_sets.append(extract_number_set(text))

    del judge_model
    torch.cuda.empty_cache()

    # Step 2: compute set metrics
    results = []
    for i in range(len(predictions)):
        pred_set = extracted_pred_sets[i]
        gold_list = json.loads(golds[i])
        gold_set = {int(float(x)) for x in gold_list}
        correct_ans = int(float(correct_answers[i]))

        # Remove the correct answer from predictions (it's not a distractor)
        pred_set.discard(correct_ans)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        union = pred_set | gold_set
        jaccard = tp / len(union) if union else 1.0

        results.append({
            "pred_set": json.dumps(sorted(pred_set)),
            "gold_set": json.dumps(sorted(gold_set)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "jaccard": jaccard,
        })

    return results


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
        description="Run inference for one (model, task, mode) and save to out/interim/."
    )
    parser.add_argument("--model-name", required=True,
                        help="HuggingFace model id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--task", required=True,
                        choices=["correct_answer", "distractor"])
    parser.add_argument("--mode", required=True, choices=["before", "after"],
                        help="'before' = base model, 'after' = fine-tuned")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to LoRA adapter directory (required when --mode after)")
    parser.add_argument("--data-csv", default=None,
                        help="Override default task CSV path")
    parser.add_argument("--out-dir", default="out/interim",
                        help="Directory to save interim result CSV")
    parser.add_argument("--suffix", default="",
                        help="Optional suffix appended to output filename")
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
        raise ValueError("No 'split' column found.")
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    if args.max_rows is not None:
        test_df = test_df.head(args.max_rows)
    print(f"Task: {args.task} | Mode: {args.mode} | Test rows: {len(test_df)}")

    # ── tokenizer ─────────────────────────────────────────────────────────
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

    # ── prepare prompts ───────────────────────────────────────────────────
    valid_mask = test_df["prompt"].notna() & test_df[target_col].notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped:
        print(f"  WARNING: dropping {n_dropped} rows with NaN prompt/target")
    test_df = test_df[valid_mask].reset_index(drop=True)

    golds = test_df[target_col].astype(str).tolist()

    if args.task == "correct_answer":
        prompts = [wrap_correct_answer_prompt(str(p)) for p in test_df["prompt"]]
    else:
        # distractor: build misconception-specific prompt per row
        prompts = [build_distractor_prompt(row) for _, row in test_df.iterrows()]

    predictions = generate_predictions(
        model, tokenizer, prompts, max_new_tokens, args.batch_size
    )

    del model
    torch.cuda.empty_cache()

    # ── score ─────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{model_short}_{args.task}_{args.mode}{args.suffix}.csv")

    if args.task == "correct_answer":
        scores = score_correct_answer_local(
            predictions, golds, batch_size=args.batch_size
        )
        result_df = pd.DataFrame(index=test_df.index)
        result_df["prediction"] = predictions
        result_df["score"] = scores
        accuracy = sum(scores) / len(scores) if scores else 0.0
        print(f"Accuracy ({args.mode}): {accuracy:.4f}  ({sum(scores)}/{len(scores)})")

    else:
        # distractor: set-level comparison
        correct_answers = test_df["correct_answer"].tolist()
        score_dicts = score_distractor_set(
            predictions, golds, correct_answers, batch_size=args.batch_size
        )
        result_df = pd.DataFrame(score_dicts)
        result_df["prediction_raw"] = predictions
        result_df["misconception_type"] = test_df["misconception_type"].tolist()

        # summary stats
        mean_jaccard = result_df["jaccard"].mean()
        total_tp = result_df["tp"].sum()
        total_fp = result_df["fp"].sum()
        total_fn = result_df["fn"].sum()
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        print(f"Distractor scores ({args.mode}):")
        print(f"  Mean Jaccard:  {mean_jaccard:.4f}")
        print(f"  Precision:     {precision:.4f}  (TP={total_tp}, FP={total_fp})")
        print(f"  Recall:        {recall:.4f}  (TP={total_tp}, FN={total_fn})")

    result_df.to_csv(out_path, index=False)
    print(f"Saved interim results -> {out_path}")


if __name__ == "__main__":
    main()

