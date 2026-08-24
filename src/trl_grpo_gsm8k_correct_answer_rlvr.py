#!/usr/bin/env python3
from __future__ import annotations

"""
Close baseline adapted from Mohammadjafari80/GSM8K-RLVR.

This is intentionally kept close to that repo's QA training setup:

- TRL GRPOTrainer
- GSM8K-style verifiable reward, no online LLM judge
- Two reward functions:
    1. format_reward_func_qa: +0.5 if completion contains "\n#### The final answer is <number>"
    2. correctness_reward_func_qa: +1.0 if answer extracted after #### matches gold
- LoRA rank 16 on q/k/v/o/up/down/gate projections
- Default model: Qwen/Qwen2.5-Math-1.5B, like the reference repo
- Default GRPO hyperparameters follow the reference repo:
    lr=2e-5
    per_device_train_batch_size=1
    gradient_accumulation_steps=8
    num_generations=6
    max_prompt_length=256
    max_completion_length=300
    epochs=2

Adaptations for your project:
- Uses your local correct_answer_pairs_gsm8k.csv by default
- Uses your existing `prompt` and `target_answer` columns
- Adds robust compatibility handling for TRL versions
- Adds optional head-only mode for speed experiments
- Adds optional Qwen chat template, but it is OFF by default to stay close to the repo
"""

import argparse
import inspect
import os
import random
import re
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


DEFAULT_CSV = (
    "/cluster/home/tunguyen1/llmgap/reasoning-efficiency/"
    "experiments/proof_search/out/correct_answer_pairs_gsm8k.csv"
)

# Reference repo default was Qwen/Qwen2.5-Math-1.5B.
# Override with --model-name Qwen/Qwen2.5-3B-Instruct if desired.
DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B"

# Close to repo's GSM8K prompt format:
# Question: ...
# Solution: Let's think step by step.
PROMPT_PREFIX = "Question: "
# PROMPT_SUFFIX = "\nSolution: Let's think step by step.\n"
PROMPT_SUFFIX = (
    "\nSolution: Let's think step by step.\n"
    "End your solution with exactly one line:\n"
    "#### The final answer is <number>\n"
)

# Close to repo's required final format:
# #### The final answer is {answer}
FORMAT_PATTERN = r"\n#### The final answer is \d+"
ANSWER_EXTRACT_PATTERN = r"####.*?([\d,]+(?:\.\d+)?)"


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        if completion and isinstance(completion[-1], list):
            return completion_to_text(completion[-1])
    return str(completion)


def extract_gold_number(gold: str) -> str:
    gold = str(gold)
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", gold)
    if nums:
        return nums[-1].replace(",", "")
    return gold.strip()


def format_reward_func_qa(completions, **kwargs):
    """Same idea as GSM8K-RLVR: +0.5 if the exact final-answer format appears."""
    completion_contents = [completion_to_text(c) for c in completions]
    matches = [re.search(FORMAT_PATTERN, content) for content in completion_contents]
    return [0.5 if match else 0.0 for match in matches]


def correctness_reward_func_qa(completions, target=None, final_answer=None, **kwargs):
    """
    Same idea as GSM8K-RLVR:
    extract answer after #### and compare numerically to ground truth.
    The original repo uses dataset field `final_answer`; this adapted file accepts
    either `target` or `final_answer`.
    """
    ground_truths = final_answer if final_answer is not None else target
    if ground_truths is None:
        raise ValueError("Need dataset column `target` or `final_answer` for correctness reward.")

    rewards = []
    for completion, ground_truth in zip(completions, ground_truths):
        completion = completion_to_text(completion)
        try:
            match = re.search(ANSWER_EXTRACT_PATTERN, completion)
            if match:
                answer = match.group(1)
                for remove_char in [",", "$", "%", "g"]:
                    answer = answer.replace(remove_char, "")
                gold = extract_gold_number(str(ground_truth))
                if abs(float(answer) - float(gold)) < 1e-3:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards


def debug_reward_func_factory(max_print: int):
    counter = {"n": 0}

    def debug_reward_func(completions, target=None, final_answer=None, prompt=None, **kwargs):
        if max_print <= 0 or counter["n"] >= max_print:
            return [0.0 for _ in completions]

        ground_truths = final_answer if final_answer is not None else target
        prompts = prompt if prompt is not None else [None] * len(completions)

        for comp, gold, pr in zip(completions, ground_truths, prompts):
            if counter["n"] >= max_print:
                break
            text = completion_to_text(comp)
            fmt = format_reward_func_qa([text])[0]
            corr = correctness_reward_func_qa([text], target=[gold])[0]
            match = re.search(ANSWER_EXTRACT_PATTERN, text)
            extracted = match.group(1) if match else None

            print("=" * 80, flush=True)
            print("PROMPT:", flush=True)
            print(pr, flush=True)
            print("COMPLETION:", flush=True)
            print(text, flush=True)
            print("GOLD:", gold, flush=True)
            print("EXTRACTED_AFTER_HASH:", extracted, flush=True)
            print("FORMAT_REWARD:", fmt, flush=True)
            print("CORRECTNESS_REWARD:", corr, flush=True)
            print("TOTAL_REWARD:", fmt + corr, flush=True)
            print("=" * 80, flush=True)
            counter["n"] += 1

        return [0.0 for _ in completions]

    return debug_reward_func


def make_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def apply_chat_template(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_dataset(args, tokenizer) -> Dataset:
    df = pd.read_csv(args.train_csv)

    required = {args.prompt_col, args.target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. Available: {list(df.columns)}")

    if args.split_col and args.split_col in df.columns:
        df = df[df[args.split_col].astype(str).str.lower() == args.train_split.lower()].reset_index(drop=True)

    df = df[df[args.prompt_col].notna() & df[args.target_col].notna()].reset_index(drop=True)

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].reset_index(drop=True)

    prompts = []
    for q in df[args.prompt_col].astype(str).tolist():
        # If your prompt column already contains only the question, this is exactly repo-like.
        # If it already contains "Question:" or "Solution:", disable wrapping with --no-repo-prompt-wrap.
        if args.no_repo_prompt_wrap:
            prompt = q
        else:
            prompt = PROMPT_PREFIX + q.strip() + PROMPT_SUFFIX

        prompt = apply_chat_template(tokenizer, prompt, args.use_chat_template)
        prompts.append(prompt)

    out = pd.DataFrame()
    out["prompt"] = prompts
    out["target"] = df[args.target_col].astype(str)
    out["final_answer"] = df[args.target_col].astype(str)

    print("=" * 80, flush=True)
    print("Dataset loaded", flush=True)
    print(f"rows={len(out)} csv={args.train_csv}", flush=True)
    print(f"repo_prompt_wrap={not args.no_repo_prompt_wrap}", flush=True)
    print(f"use_chat_template={args.use_chat_template}", flush=True)
    if len(out):
        print("\nExample prompt:", flush=True)
        print(out.iloc[0]["prompt"][:2000], flush=True)
        print("\nExample final_answer:", out.iloc[0]["final_answer"], flush=True)
    print("=" * 80, flush=True)

    return Dataset.from_pandas(out, preserve_index=False)


def configure_head_only(model, train_embed_too: bool):
    for _, p in model.named_parameters():
        p.requires_grad = False

    trainable_modules = []
    if hasattr(model, "lm_head") and model.lm_head is not None:
        for p in model.lm_head.parameters():
            p.requires_grad = True
        trainable_modules.append("lm_head")
    else:
        out_emb = model.get_output_embeddings()
        if out_emb is not None:
            for p in out_emb.parameters():
                p.requires_grad = True
            trainable_modules.append("output_embeddings")

    if train_embed_too:
        in_emb = model.get_input_embeddings()
        if in_emb is not None:
            for p in in_emb.parameters():
                p.requires_grad = True
            trainable_modules.append("input_embeddings")

    print_trainable_parameters(model)
    print("Head-only trainable modules:", trainable_modules, flush=True)


def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}", flush=True)
    print(f"All parameters: {all_params}", flush=True)
    print(f"Percentage of trainable parameters: {100 * trainable_params / max(all_params, 1):.2f}%", flush=True)


def build_lora_config(args):
    if args.train_mode != "lora":
        return None
    if LoraConfig is None:
        raise ImportError("peft is required for LoRA.")
    rank = args.lora_r
    return LoraConfig(
        r=rank,
        lora_alpha=args.lora_alpha if args.lora_alpha is not None else rank * 2,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        lora_dropout=args.lora_dropout,
    )


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
    )

    if args.load_in_4bit:
        model_kwargs["quantization_config"] = make_quant_config(True)
        model_kwargs["torch_dtype"] = None
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.train_mode == "head":
        configure_head_only(model, args.train_embed_too)
    elif args.train_mode == "lora":
        peft_config = build_lora_config(args)
        if args.attach_lora_before_trainer:
            if get_peft_model is None:
                raise ImportError("get_peft_model unavailable.")
            model = get_peft_model(model, peft_config)
            print_trainable_parameters(model)
            peft_config = None
        return model, tokenizer, peft_config
    else:
        print_trainable_parameters(model)

    return model, tokenizer, None


def build_grpo_config(args):
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    desired = dict(
        output_dir=args.out_dir,
        run_name=args.run_name,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        bf16=bf16,
        fp16=fp16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        log_on_each_node=False,
        remove_unused_columns=False,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    sig = inspect.signature(GRPOConfig.__init__)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in desired.items() if k in accepted}
    dropped = sorted(k for k in desired if k not in accepted)

    print("GRPOConfig accepted keys:", sorted(filtered.keys()), flush=True)
    if dropped:
        print("GRPOConfig dropped keys:", dropped, flush=True)

    cfg = GRPOConfig(**filtered)

    # Compatibility for TRL versions that do not accept some fields in constructor.
    for field in ["num_generations", "max_prompt_length", "max_completion_length", "beta", "temperature", "top_p"]:
        if hasattr(cfg, field):
            setattr(cfg, field, getattr(args, field))

    return cfg


def parse_args():
    p = argparse.ArgumentParser("GSM8K-RLVR close baseline adapted to local CSV")

    p.add_argument("--train-csv", default=DEFAULT_CSV)
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default=None)

    p.add_argument("--prompt-col", default="prompt")
    p.add_argument("--target-col", default="target_answer")
    p.add_argument("--split-col", default="split")
    p.add_argument("--train-split", default="train")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train-mode", choices=["lora", "head", "full"], default="lora")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--attach-lora-before-trainer", action="store_true", default=True)
    p.add_argument("--trainer-peft-config", dest="attach_lora_before_trainer", action="store_false")

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")

    p.add_argument("--train-embed-too", action="store_true")
    p.add_argument("--attn-implementation", default="flash_attention_2")

    p.add_argument("--num-generations", type=int, default=6)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-prompt-length", type=int, default=256)
    p.add_argument("--max-completion-length", type=int, default=300)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-grad-norm", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)

    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--report-to", default="none")
    p.add_argument("--debug-print-rewards", type=int, default=20)

    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--no-repo-prompt-wrap", action="store_true")

    args = p.parse_args()

    if args.run_name is None:
        args.run_name = f"GRPO-GSM8K-qa-{args.model_name.split('/')[-1]}"

    if args.per_device_train_batch_size % args.num_generations != 0:
        print(
            "WARNING: Your TRL version may require per_device_train_batch_size divisible by num_generations. "
            "The reference repo uses batch_size=1 and num_generations=6, but newer TRL versions may reject this. "
            "If it errors, run with --per-device-train-batch-size 6 --gradient-accumulation-steps 8.",
            flush=True,
        )

    if args.report_to == "none":
        args.report_to = []

    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer, peft_config = load_model_and_tokenizer(args)
    dataset = build_dataset(args, tokenizer)
    training_args = build_grpo_config(args)

    rewards_funcs = [
        format_reward_func_qa,
        correctness_reward_func_qa,
        debug_reward_func_factory(args.debug_print_rewards),
    ]

    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rewards_funcs,
        args=training_args,
        train_dataset=dataset,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    try:
        trainer = GRPOTrainer(**trainer_kwargs)
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        trainer = GRPOTrainer(**trainer_kwargs)

    print(
        f"trainer.args.max_completion_length actual = {getattr(trainer.args, 'max_completion_length', None)}",
        flush=True,
    )

    trainer.train()
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved model/adapters to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()