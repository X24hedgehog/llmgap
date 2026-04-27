#!/usr/bin/env python3
"""Fine-tune a HuggingFace Instruct model on one task using LoRA + SFTTrainer.

Tasks:
  correct_answer    — input: local subproblem text, output: numeric answer
  next_subquestion  — input: partial solution state, output: next subquestion text

Example:
    python finetune.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --task correct_answer \\
        --out-dir out/checkpoints/Qwen2.5-0.5B-Instruct_correct_answer

Output:
    out/checkpoints/<name>_<task>/final/   (merged / unmerged LoRA weights)
    out/checkpoints/<name>_<task>/run_config.json
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer

# Try new-style SFTConfig (trl >= 0.9), fall back to TrainingArguments
try:
    from trl import SFTConfig as _SFTConfigCls
    _USE_SFT_CONFIG = True
except ImportError:
    from transformers import TrainingArguments as _SFTConfigCls
    _USE_SFT_CONFIG = False

# ── constants ─────────────────────────────────────────────────────────────────

TASK_TARGET_COL = {
    "correct_answer": "target_question_reasoning_trace",  # subtree CoT, built by generate_target_reasoning.py
    "next_subquestion": "next_subquestion",
    "distractor": "target_distractor_reasoning_traces",  # JSON list of erroneous CoTs
}

TASK_CSV = {
    "correct_answer": "out/correct_answer_pairs.csv",
    "next_subquestion": "out/next_subquestion_pairs.csv",
    "distractor": "out/distractor_pairs.csv",
}

# Attention-projection modules to inject LoRA into (works for Qwen2, LLaMA, Gemma)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Models that need --load-in-4bit by default (>= 6 B parameters makes 40 GB tight)
LARGE_MODELS = {"7b", "8b"}


def _needs_4bit(model_name: str) -> bool:
    lower = model_name.lower()
    return any(tag in lower for tag in LARGE_MODELS)


# ── dataset builder ───────────────────────────────────────────────────────────

def _format_chat(tokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": str(response)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


CORRECT_ANSWER_INSTRUCTION = "\n\nSolve this problem step by step."
DISTRACTOR_INSTRUCTION = "\n\nIncorrect Answer: Let's think step by step."


def build_hf_dataset(df: pd.DataFrame, tokenizer, target_col: str, task: str) -> Dataset:
    texts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Formatting dataset"):
        if task == "correct_answer":
            prompt = str(row["prompt"]).rstrip() + CORRECT_ANSWER_INSTRUCTION
        elif task == "distractor":
            prompt = str(row["prompt"]).rstrip() + DISTRACTOR_INSTRUCTION
        else:
            prompt = str(row["prompt"])

        if task == "distractor":
            # target_col is a JSON list of reasoning traces; create one
            # training example per trace so the model sees all distractors.
            import json as _json
            try:
                rts = _json.loads(row[target_col])
            except (ValueError, TypeError):
                rts = [row[target_col]]
            for rt in rts:
                texts.append(_format_chat(tokenizer, prompt, str(rt)))
        else:
            texts.append(_format_chat(tokenizer, prompt, str(row[target_col])))

    return Dataset.from_dict({"text": texts})


# ── training ──────────────────────────────────────────────────────────────────

def train(args) -> None:
    target_col = TASK_TARGET_COL[args.task]

    df = pd.read_csv(args.train_csv)
    if "split" in df.columns:
        df = df[df["split"] == "train"].reset_index(drop=True)

    # Fallback: if target column is null/empty, use reasoning_trace instead
    if target_col == "target_question_reasoning_trace" and "reasoning_trace" in df.columns:
        mask = df[target_col].isna() | (df[target_col].astype(str).str.strip() == "")
        df.loc[mask, target_col] = df.loc[mask, "reasoning_trace"]
        if mask.any():
            print(f"  Fell back to 'reasoning_trace' for {mask.sum()} rows with empty '{target_col}'")

    print(f"Training rows: {len(df)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── quantization (optional) ───────────────────────────────────────────
    load_4bit = args.load_in_4bit or _needs_4bit(args.model_name)
    quant_config = None
    if load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # ── model ─────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if not load_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── dataset ───────────────────────────────────────────────────────────
    dataset = build_hf_dataset(df, tokenizer, target_col, task=args.task)

    # ── training config ───────────────────────────────────────────────────
    training_kwargs = dict(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=5,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    if _USE_SFT_CONFIG:
        # max_seq_length was renamed to max_length in some TRL versions
        import inspect as _inspect
        _sft_params = _inspect.signature(_SFTConfigCls.__init__).parameters
        _seq_len_kwarg = (
            "max_seq_length" if "max_seq_length" in _sft_params
            else "max_length" if "max_length" in _sft_params
            else None
        )
        if _seq_len_kwarg:
            training_kwargs[_seq_len_kwarg] = args.max_seq_len
        training_config = _SFTConfigCls(
            **training_kwargs,
            dataset_text_field="text",
        )
        trainer = SFTTrainer(
            model=model,
            args=training_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:
        # Older TRL: pass extra args directly to SFTTrainer
        training_config = _SFTConfigCls(**training_kwargs)
        trainer = SFTTrainer(
            model=model,
            args=training_config,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_len,
            dataset_text_field="text",
        )

    # Auto-resume from the latest epoch checkpoint if one exists (handles
    # network drops / node failures without restarting from scratch).
    resume = any(
        d.name.startswith("checkpoint-")
        for d in Path(args.out_dir).iterdir()
        if d.is_dir()
    ) if Path(args.out_dir).exists() else False
    if resume:
        print(f"Resuming from latest checkpoint in {args.out_dir}")
    trainer.train(resume_from_checkpoint=resume)

    # Save final adapter + tokenizer directly into out_dir
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved fine-tuned model → {args.out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune with LoRA on a single task.")
    parser.add_argument("--model-name", required=True,
                        help="HuggingFace model id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--task", required=True,
                        choices=["correct_answer", "next_subquestion", "distractor"])
    parser.add_argument("--train-csv", default=None,
                        help="Path to training CSV (default: out/splits/<task>_train.csv)")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to save checkpoints and final model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha (default: 16)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Force 4-bit quantization (auto-enabled for 7B+ models)")
    args = parser.parse_args()

    if args.train_csv is None:
        args.train_csv = TASK_CSV[args.task]

    train(args)


if __name__ == "__main__":
    main()
