#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# ---------------------------------------------------------------------
# Compatibility patch:
# Some TRL versions expect torch.distributed.fsdp.FSDPModule, but some
# PyTorch 2.5 builds do not expose it. We are not using FSDP here, so
# provide a harmless fallback before importing TRL.
# ---------------------------------------------------------------------
try:
    import torch.distributed.fsdp as _fsdp
    if not hasattr(_fsdp, "FSDPModule"):
        if hasattr(_fsdp, "FullyShardedDataParallel"):
            _fsdp.FSDPModule = _fsdp.FullyShardedDataParallel
        else:
            class _DummyFSDPModule:
                pass
            _fsdp.FSDPModule = _DummyFSDPModule
except Exception:
    pass

from trl import GRPOConfig, GRPOTrainer


PROMPT_TEMPLATE = """Solve the following grade school math problem step by step.

Rules:
- Be concise.
- The last line must be exactly: #### <number>
- Do not write anything after the final answer line.

Question: {question}
Answer:"""


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))

        if completion and isinstance(completion[-1], list):
            return completion_to_text(completion[-1])

    return str(completion)


def extract_last_number(text: str) -> str | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return nums[-1] if nums else None


def extract_final_answer(text: str) -> str:
    text = str(text).replace(",", "")

    marker = re.search(r"####\s*\$?\s*(-?\d+(?:\.\d+)?)", text)
    if marker:
        return marker.group(1)

    boxed = re.search(r"\\boxed\{\s*\$?\s*(-?\d+(?:\.\d+)?)\s*\}", text)
    if boxed:
        return boxed.group(1)

    final_phrase = re.search(
        r"final answer (?:is|:)\s*\$?\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if final_phrase:
        return final_phrase.group(1)

    last_number = extract_last_number(text)
    if last_number is not None:
        return last_number

    return text.strip()


def normalize_answer(answer: str) -> str:
    answer = str(answer).strip().replace(",", "").replace("$", "")

    marker = re.search(r"####\s*\$?\s*(-?\d+(?:\.\d+)?)", answer)
    if marker:
        answer = marker.group(1)

    last_number = extract_last_number(answer)
    if last_number is not None:
        answer = last_number

    try:
        value = float(answer)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return answer.strip()


def answers_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def make_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )



def configure_trainable_parameters(model, args) -> None:
    """
    Optional head-only RL mode.

    If args.train_lm_head_only is enabled:
    - freeze the whole transformer backbone
    - train only the final language-model head
    - optionally also train token embeddings with --train-embed-too

    This is much cheaper than LoRA/full fine-tuning, but less expressive.
    It is useful as a fast RL baseline.
    """
    if not getattr(args, "train_lm_head_only", False):
        return

    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_names = []

    # Main output projection of causal LM.
    if hasattr(model, "lm_head") and model.lm_head is not None:
        for param in model.lm_head.parameters():
            param.requires_grad = True
        trainable_names.append("lm_head")

    # Optional: train input embeddings too.
    if getattr(args, "train_embed_too", False):
        try:
            emb = model.get_input_embeddings()
            if emb is not None:
                for param in emb.parameters():
                    param.requires_grad = True
                trainable_names.append("input_embeddings")
        except Exception as exc:
            print(f"Could not enable input embedding training: {exc}", flush=True)

    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        n = param.numel()
        total_params += n
        if param.requires_grad:
            trainable_params += n

    print("=" * 80, flush=True)
    print("HEAD-ONLY TRAINING MODE ENABLED", flush=True)
    print("Trainable modules:", trainable_names, flush=True)
    print(f"Trainable parameters: {trainable_params:,}", flush=True)
    print(f"Total parameters:     {total_params:,}", flush=True)
    print(f"Trainable fraction:   {100.0 * trainable_params / max(total_params, 1):.6f}%", flush=True)
    print("=" * 80, flush=True)

def build_dataset(args: argparse.Namespace) -> Dataset:
    df = pd.read_csv(args.train_csv)

    required = {
        args.question_col,
        args.answer_col,
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available columns: {list(df.columns)}"
        )

    if args.split_col and args.split_col in df.columns:
        df = df[
            df[args.split_col].astype(str).str.lower()
            == args.train_split.lower()
        ].reset_index(drop=True)

    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].reset_index(drop=True)

    df = df.dropna(
        subset=[
            args.question_col,
            args.answer_col,
        ]
    ).reset_index(drop=True)

    out = pd.DataFrame()

    # Your dataset also has a "prompt" column, but it is the raw math question.
    # We wrap the question with an instruction prompt for RL generation.
    questions = df[args.question_col].astype(str).tolist()

    out["prompt"] = [
        PROMPT_TEMPLATE.format(question=q)
        for q in questions
    ]

    out["question"] = df[args.question_col].astype(str)
    out["target_answer"] = df[args.answer_col].astype(str)

    print("Loaded correct-answer dataset", flush=True)
    print(f"  path: {args.train_csv}", flush=True)
    print(f"  rows: {len(out)}", flush=True)
    print(f"  columns used: {list(out.columns)}", flush=True)

    if len(out) > 0:
        print("\nExample prompt:", flush=True)
        print(out.iloc[0]["prompt"], flush=True)
        print("\nExample target_answer:", flush=True)
        print(out.iloc[0]["target_answer"], flush=True)

    return Dataset.from_pandas(out, preserve_index=False)


def load_policy_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = make_quant_config(args.load_in_4bit)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRL GRPO for GSM8K correct-answer RLVR with binary correctness reward."
    )

    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--question-col", default="question")
    parser.add_argument("--answer-col", default="target_answer")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--max-rows", type=int, default=None)

    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=128)

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)

    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)

    # Optional auxiliary format reward.
    # If set to 0, reward is purely binary correctness.
    parser.add_argument("--format-reward-weight", type=float, default=0.0)

    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument(
        "--train-lm-head-only",
        action="store_true",
        help="Freeze the transformer backbone and train only the final lm_head. Disables PEFT/LoRA.",
    )
    parser.add_argument(
        "--train-embed-too",
        action="store_true",
        help="With --train-lm-head-only, also train input token embeddings.",
    )

    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--debug-print-rewards", type=int, default=0)

    return parser.parse_args()


def build_grpo_config(args: argparse.Namespace) -> GRPOConfig:
    """
    Build GRPOConfig in a TRL-version-adaptive way.

    Your local TRL version does not accept all fields used by some examples,
    e.g. max_prompt_length. We inspect the local GRPOConfig signature and
    only pass supported kwargs.
    """
    import inspect

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    desired_kwargs = {
        "output_dir": args.out_dir,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "report_to": [],
        "remove_unused_columns": False,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    sig = inspect.signature(GRPOConfig.__init__)
    accepted = set(sig.parameters.keys())

    filtered_kwargs = {
        k: v for k, v in desired_kwargs.items()
        if k in accepted
    }

    dropped = {
        k: v for k, v in desired_kwargs.items()
        if k not in accepted
    }

    print("\\nGRPOConfig accepted keys:", sorted(filtered_kwargs.keys()), flush=True)
    if dropped:
        print("GRPOConfig dropped unsupported keys:", sorted(dropped.keys()), flush=True)

    config = GRPOConfig(**filtered_kwargs)

    post_init_fields = {
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
    }

    for field, value in post_init_fields.items():
        if hasattr(config, field):
            setattr(config, field, value)
            print(f"Set GRPOConfig.{field} = {value}", flush=True)

    return config

def build_peft_config(args: argparse.Namespace):
    if getattr(args, "train_lm_head_only", False):
        print("Head-only mode enabled: disabling PEFT/LoRA.", flush=True)
        return None

    if not args.use_peft:
        return None

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = build_dataset(args)

    model, tokenizer = load_policy_and_tokenizer(args)

    if args.gradient_checkpointing:
        if getattr(args, "train_lm_head_only", False):
            print(
                "Warning: gradient checkpointing is usually unnecessary in head-only mode, "
                "because the transformer backbone is frozen.",
                flush=True,
            )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    debug_counter = {"n": 0}

    def correctness_reward(completions, target_answer=None, **kwargs):
        if target_answer is None:
            raise ValueError(
                "Reward function expected dataset column `target_answer`. "
                "Check build_dataset() and remove_unused_columns=False."
            )

        rewards = []

        for completion, gold in zip(completions, target_answer):
            text = completion_to_text(completion)
            pred = extract_final_answer(text)
            reward = 1.0 if answers_match(pred, str(gold)) else 0.0
            rewards.append(reward)

            if args.debug_print_rewards > 0 and debug_counter["n"] < args.debug_print_rewards:
                print("=" * 80, flush=True)
                print("COMPLETION:", flush=True)
                print(text, flush=True)
                print("EXTRACTED:", pred, flush=True)
                print("GOLD:", str(gold), flush=True)
                print("REWARD:", reward, flush=True)
                print("=" * 80, flush=True)
                debug_counter["n"] += 1

        return rewards

    def format_reward(completions, **kwargs):
        rewards = []

        for completion in completions:
            text = completion_to_text(completion).replace(",", "")
            ok = bool(re.search(r"####\s*\$?\s*-?\d+(?:\.\d+)?", text))
            rewards.append(args.format_reward_weight if ok else 0.0)

        return rewards

    reward_funcs = [correctness_reward]
    if args.format_reward_weight > 0:
        reward_funcs.append(format_reward)

    peft_config = build_peft_config(args)
    training_args = build_grpo_config(args)

    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise

        print(
            "GRPOTrainer does not accept processing_class; falling back to tokenizer=tokenizer",
            flush=True,
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print(f"Saved TRL GRPO correct-answer model to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()