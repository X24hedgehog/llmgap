#!/usr/bin/env python3
from __future__ import annotations

"""
Evaluate GSM8K correct-answer performance for:
1. base model
2. optional LoRA/PEFT adapter trained with GRPO/RLVR

It uses the same prompt style as the successful RLVR training run:
- Qwen chat template optional
- final line instruction:
  #### The final answer is <number>

Metrics:
- strict_acc: correct number extracted after #### The final answer is ...
- loose_acc: correct number extracted from ####, boxed answer, final-answer phrase, or last number
- format_rate: output ends with the required final answer line
- avg_gen_len
- clipped_rate proxy based on generated token count == max_new_tokens
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


DEFAULT_CSV = (
    "/cluster/home/tunguyen1/llmgap/reasoning-efficiency/"
    "experiments/proof_search/out/correct_answer_pairs_gsm8k.csv"
)

PROMPT_SUFFIX = (
    "\nSolution: Let's think step by step.\n"
    "End your solution with exactly one line:\n"
    "#### The final answer is <number>"
)

STRICT_RE = re.compile(r"####\s*The final answer is\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)
STRICT_FINAL_LINE_RE = re.compile(
    r"####\s*The final answer is\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
FINAL_PHRASE_RE = re.compile(
    r"(?:final answer|answer)(?:\s+is)?[:\s]*\$?\s*(-?[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def make_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def normalize_num(x: str | None) -> str | None:
    if x is None:
        return None
    return str(x).replace(",", "").strip()


def extract_gold(gold: str) -> str | None:
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", str(gold))
    if nums:
        return normalize_num(nums[-1])
    return normalize_num(str(gold))


def extract_strict(text: str) -> str | None:
    m = STRICT_RE.search(str(text))
    if not m:
        return None
    return normalize_num(m.group(1))


def has_strict_final_line(text: str) -> bool:
    return STRICT_FINAL_LINE_RE.search(str(text).strip()) is not None


def extract_loose(text: str) -> str | None:
    text = str(text)

    strict = extract_strict(text)
    if strict is not None:
        return strict

    boxed = BOXED_RE.search(text)
    if boxed:
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", boxed.group(1))
        if nums:
            return normalize_num(nums[-1])

    phrase = FINAL_PHRASE_RE.search(text)
    if phrase:
        return normalize_num(phrase.group(1))

    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if nums:
        return normalize_num(nums[-1])

    return None


def nums_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return str(a).strip() == str(b).strip()


def make_prompt(tokenizer, question_or_prompt: str, use_chat_template: bool, repo_prompt_wrap: bool) -> str:
    text = str(question_or_prompt).strip()

    if repo_prompt_wrap:
        if text.lower().lstrip().startswith("question:"):
            prompt = text + PROMPT_SUFFIX
        else:
            prompt = "Question: " + text + PROMPT_SUFFIX
    else:
        prompt = text + PROMPT_SUFFIX

    if use_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt + "\n"


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=make_quant_config(args.load_in_4bit),
        torch_dtype=None if args.load_in_4bit else torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True

    if args.adapter_path:
        if PeftModel is None:
            raise ImportError("peft is required to load --adapter-path.")
        print(f"Loading LoRA adapter from: {args.adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter_path)

        if args.merge_adapter:
            print("Merging adapter into base model for evaluation...", flush=True)
            model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str], args):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    outputs = model.generate(**gen_kwargs)

    new_token_ids = outputs[:, input_len:]
    texts = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
    lengths = [int((row != tokenizer.pad_token_id).sum().item()) for row in new_token_ids]
    return texts, lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--train-csv", default=DEFAULT_CSV)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--prompt-col", default="prompt")
    parser.add_argument("--target-col", default="target_answer")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--merge-adapter", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--no-repo-prompt-wrap", dest="repo_prompt_wrap", action="store_false")
    parser.set_defaults(repo_prompt_wrap=True)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    if args.split_col in df.columns:
        df = df[df[args.split_col].astype(str).str.lower() == args.eval_split.lower()].reset_index(drop=True)

    df = df[df[args.prompt_col].notna() & df[args.target_col].notna()].reset_index(drop=True)

    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].reset_index(drop=True)

    print("=" * 80, flush=True)
    print("Evaluation setup", flush=True)
    print(f"model_name: {args.model_name}", flush=True)
    print(f"adapter_path: {args.adapter_path}", flush=True)
    print(f"csv: {args.train_csv}", flush=True)
    print(f"split: {args.eval_split}", flush=True)
    print(f"rows: {len(df)}", flush=True)
    print(f"use_chat_template: {args.use_chat_template}", flush=True)
    print(f"repo_prompt_wrap: {args.repo_prompt_wrap}", flush=True)
    print(f"max_new_tokens: {args.max_new_tokens}", flush=True)
    print("=" * 80, flush=True)

    model, tokenizer = load_model_and_tokenizer(args)

    prompts = [
        make_prompt(tokenizer, q, args.use_chat_template, args.repo_prompt_wrap)
        for q in df[args.prompt_col].astype(str).tolist()
    ]
    golds = [extract_gold(x) for x in df[args.target_col].astype(str).tolist()]

    all_rows = []
    for start in tqdm(range(0, len(df), args.batch_size)):
        batch_prompts = prompts[start:start + args.batch_size]
        texts, gen_lens = generate_batch(model, tokenizer, batch_prompts, args)

        for i, (pred_text, gen_len) in enumerate(zip(texts, gen_lens)):
            idx = start + i
            strict_pred = extract_strict(pred_text)
            loose_pred = extract_loose(pred_text)
            gold = golds[idx]

            all_rows.append({
                "idx": idx,
                "gold": gold,
                "strict_pred": strict_pred,
                "loose_pred": loose_pred,
                "strict_correct": nums_equal(strict_pred, gold),
                "loose_correct": nums_equal(loose_pred, gold),
                "has_required_final_line": has_strict_final_line(pred_text),
                "gen_len": gen_len,
                "clipped": gen_len >= args.max_new_tokens,
                "prompt": df.iloc[idx][args.prompt_col],
                "completion": pred_text,
            })

    out = pd.DataFrame(all_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    metrics = {
        "n": len(out),
        "strict_acc": float(out["strict_correct"].mean()) if len(out) else 0.0,
        "loose_acc": float(out["loose_correct"].mean()) if len(out) else 0.0,
        "format_rate": float(out["has_required_final_line"].mean()) if len(out) else 0.0,
        "avg_gen_len": float(out["gen_len"].mean()) if len(out) else 0.0,
        "clipped_rate": float(out["clipped"].mean()) if len(out) else 0.0,
    }

    metrics_path = str(Path(args.out_csv).with_suffix(".metrics.txt"))
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print("=" * 80, flush=True)
    print("METRICS", flush=True)
    for k, v in metrics.items():
        print(f"{k}: {v}", flush=True)
    print(f"Saved predictions to: {args.out_csv}", flush=True)
    print(f"Saved metrics to: {metrics_path}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()