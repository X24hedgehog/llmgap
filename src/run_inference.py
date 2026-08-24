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
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openai_judge import DEFAULT_OPENAI_EEDI_JUDGE_MODEL, judge_yes_no_openai
from prompt import (
    build_answer_equivalence_prompt,
)

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

EEDI_PROMPT_STYLES = {"eedi_correct_answer", "eedi_distractor"}

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
    prompt_style = str(row.get("prompt_style", ""))
    if prompt_style == "eedi_correct_answer":
        return f"Question: {str(row['question']).strip()}" + CORRECT_ANSWER_INSTRUCTION
    if prompt_style == "eedi_distractor":
        return (
            "You are solving a math question as a student with the following "
            f"misconception: {str(row['misconception_name']).strip()}\n\n"
            f"Question: {str(row['question']).strip()}\n\n"
            "Think step by step and give the incorrect answer this student "
            "would produce."
        )

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
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=judge_tokenizer.pad_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for seq in out:
            text = judge_tokenizer.decode(
                seq[input_len:], skip_special_tokens=True
            ).strip()
            verdicts = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
            final_verdict = verdicts[-1].lower() if verdicts else "no"
            scores.append(1 if final_verdict == "yes" else 0)
    return scores


def _parse_distractor_gold(gold_str: str):
    import ast

    parsed = ast.literal_eval(gold_str)
    return parsed if isinstance(parsed, list) else [parsed]


def _is_numeric_like(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?((\d+\.?\d*)|(\.\d+))", str(text).strip()))


def _all_numeric_distractor_golds(golds) -> bool:
    for gold_str in golds:
        gold_values = _parse_distractor_gold(gold_str)
        if not gold_values or not all(_is_numeric_like(value) for value in gold_values):
            return False
    return True


def _use_semantic_distractor_scoring(prompt_style: str) -> bool:
    return str(prompt_style) == "eedi_distractor"


def _use_semantic_correct_answer_scoring(prompt_style: str) -> bool:
    return str(prompt_style) == "eedi_correct_answer"


def score_correct_answer(predictions, golds, prompts, prompt_styles, batch_size=8):
    """Score correct answers.

    Numeric golds use the existing judge-based solver check. Non-numeric golds,
    such as EEDI answer-option text, use normalized exact match unless the row is
    in the EEDI setting, in which case OpenAI handles semantic equivalence.
    """

    def normalize_text(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"^(answer:|correct answer:)\s*", "", text)
        text = text.replace("\\(", "").replace("\\)", "")
        text = text.replace("\\[", "").replace("\\]", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def is_numeric_like(text: str) -> bool:
        return bool(re.fullmatch(r"[-+]?((\d+\.?\d*)|(\.\d+))", str(text).strip()))

    scores = [None] * len(predictions)
    numeric_examples = []
    semantic_examples = []

    for idx, (prediction, gold, problem_context, prompt_style) in enumerate(
        zip(predictions, golds, prompts, prompt_styles)
    ):
        if _use_semantic_correct_answer_scoring(prompt_style):
            semantic_examples.append((idx, problem_context, prediction, gold))
            continue

        if is_numeric_like(gold):
            numeric_examples.append((idx, prediction, gold))
            continue

        scores[idx] = 1 if normalize_text(prediction) == normalize_text(gold) else 0

    if numeric_examples:
        template = (
            "A student solved the following math problem and wrote this solution:\n"
            "{prediction}\n\n"
            "The correct final answer is: {gold}\n\n"
            "Did the student arrive at the correct final answer? "
            "Answer only 'yes' or 'no'."
        )
        judge_model, judge_tok = _load_judge()
        judge_prompts = [
            template.format(prediction=prediction, gold=gold)
            for _, prediction, gold in numeric_examples
        ]
        numeric_scores = _judge_yes_no(judge_model, judge_tok, judge_prompts, batch_size)
        del judge_model
        torch.cuda.empty_cache()
        for (row_idx, _, _), score in zip(numeric_examples, numeric_scores):
            scores[row_idx] = score

    if semantic_examples:
        semantic_prompts = [
            build_answer_equivalence_prompt(
                problem_context,
                normalize_text(prediction),
                normalize_text(gold),
            )
            for _, problem_context, prediction, gold in semantic_examples
        ]
        print(f"Using OpenAI EEDI judge model: {DEFAULT_OPENAI_EEDI_JUDGE_MODEL}")
        semantic_scores = judge_yes_no_openai(semantic_prompts, batch_size=batch_size)
        for (row_idx, _, _, _), score in zip(semantic_examples, semantic_scores):
            scores[row_idx] = score

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


def score_distractor(predictions, golds, prompts, prompt_styles, batch_size=8):
    """Score distractor predictions for both numeric and answer-text outputs."""

    def normalize_text(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"^(distractor\s*\d*\s*:|incorrect student answer:|answer:)\s*", "", text)
        text = text.replace("\\(", "").replace("\\)", "")
        text = text.replace("\\[", "").replace("\\]", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    scores = []
    semantic_examples = []

    for problem_context, prompt_style, pred, gold_str in zip(
        prompts, prompt_styles, predictions, golds
    ):
        gold_values = _parse_distractor_gold(gold_str)
        if not _use_semantic_distractor_scoring(prompt_style):
            gold_nums = {int(float(value)) for value in gold_values}
            all_nums = re.findall(r'-?\b\d+\b', pred)
            if all_nums:
                scores.append(1 if int(all_nums[-1]) in gold_nums else 0)
            else:
                scores.append(0)
            continue

        semantic_examples.append((len(scores), problem_context, pred, gold_values))
        scores.append(None)

    if not semantic_examples:
        return scores

    expanded_prompts = []
    row_slices = []
    for row_idx, problem_context, pred, gold_values in semantic_examples:
        start_idx = len(expanded_prompts)
        pred_norm = normalize_text(pred)
        for gold in gold_values:
            expanded_prompts.append(
                build_answer_equivalence_prompt(
                    problem_context,
                    pred_norm,
                    normalize_text(gold),
                )
            )
        row_slices.append((row_idx, start_idx, len(expanded_prompts)))

    print(f"Using OpenAI EEDI judge model: {DEFAULT_OPENAI_EEDI_JUDGE_MODEL}")
    expanded_scores = judge_yes_no_openai(expanded_prompts, batch_size=batch_size)

    for row_idx, start, end in row_slices:
        scores[row_idx] = 1 if any(expanded_scores[start:end]) else 0

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
    prompt_styles = (
        test_df["prompt_style"].astype(str).tolist()
        if "prompt_style" in test_df.columns
        else [""] * len(test_df)
    )
    golds = test_df[target_col].astype(str).tolist()

    predictions = generate_predictions(
        model, tokenizer, prompts, max_new_tokens, args.batch_size
    )

    del model
    torch.cuda.empty_cache()

    # ── score ─────────────────────────────────────────────────────────────
    if args.task == "correct_answer":
        scores = score_correct_answer(
            predictions,
            golds,
            prompts,
            prompt_styles,
            args.batch_size,
        )
    elif args.task == "next_subquestion":
        scores = score_next_subquestion(predictions, golds, args.batch_size)
    elif args.task == "distractor":
        scores = score_distractor(
            predictions,
            golds,
            prompts,
            prompt_styles,
            args.batch_size,
        )

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
