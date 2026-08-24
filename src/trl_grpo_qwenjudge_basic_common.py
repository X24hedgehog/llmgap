#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import os
import re
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------
# FSDP compatibility patch for some TRL / torch combinations
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

try:
    from peft import LoraConfig, TaskType
except Exception:
    LoraConfig = None
    TaskType = None


# ---------------------------------------------------------------------
# Task conventions, aligned with run_inference.py
# ---------------------------------------------------------------------

DEFAULT_POLICY_MODEL = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

CORRECT_ANSWER_INSTRUCTION = "\n\nSolve this problem step by step."
DISTRACTOR_INSTRUCTION = "\n\nIncorrect Answer: Let's think step by step."

TASK_TARGET_COL = {
    "correct_answer": "target_answer",
    "next_subquestion": "next_subquestion",
    "distractor": "target_distractor_answers",
}

TASK_DEFAULT_CSV = {
    "correct_answer": "/cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs_gsm8k.csv",
    "next_subquestion": "/cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs_gsm8k.csv",
    "distractor": "/cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/distractor_pairs_gsm8k.csv",
}

# Practical defaults. You can increase correct_answer/distractor to 512 later.
TASK_MAX_COMPLETION_LENGTH = {
    "correct_answer": 256,
    "next_subquestion": 80,
    "distractor": 256,
}


def wrap_task_prompt(task: str, prompt: str) -> str:
    prompt = str(prompt).rstrip()

    if task == "correct_answer":
        return prompt + CORRECT_ANSWER_INSTRUCTION

    if task == "distractor":
        return prompt + DISTRACTOR_INSTRUCTION

    if task == "next_subquestion":
        return prompt

    raise ValueError(f"Unknown task: {task}")


def apply_policy_chat_template(tokenizer, user_prompt: str, use_chat_template: bool = True) -> str:
    if not use_chat_template:
        return str(user_prompt)

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": str(user_prompt)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        if completion and isinstance(completion[-1], list):
            return completion_to_text(completion[-1])

    return str(completion)


def parse_yes_no(text: str) -> int:
    """
    Match run_inference.py-style logic: generated answer should start with yes.
    """
    text = str(text).strip().lower()
    return 1 if text.startswith("yes") else 0


def make_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


class QwenYesNoJudge:
    """
    Frozen Qwen yes/no judge.

    Uses the same judge prompts as reasoning-efficiency run_inference.py:
    - correct_answer: did student arrive at gold final answer?
    - next_subquestion: are two subquestions semantically equivalent?
    - distractor: did student arrive at any expected distractor answer?
    """

    def __init__(
        self,
        model_name: str = JUDGE_MODEL_NAME,
        load_in_4bit: bool = True,
        max_input_length: int = 1024,
        max_new_tokens: int = 4,
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens

        print(f"Loading judge model: {model_name}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        quant_config = make_quant_config(load_in_4bit)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=None if load_in_4bit else torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def build_prompt(self, task: str, prediction: str, gold: str) -> str:
        if task == "correct_answer":
            return (
                "A student solved the following math problem and wrote this solution:\n"
                f"{prediction}\n\n"
                f"The correct final answer is: {gold}\n\n"
                "Did the student arrive at the correct final answer? "
                "Answer only 'yes' or 'no'."
            )

        if task == "distractor":
            return (
                "A student was asked to produce an incorrect answer (a distractor) "
                "for a math problem and wrote the following reasoning:\n"
                f"{prediction}\n\n"
                f"The expected distractor answers are: {gold}\n\n"
                "Did the student arrive at any of the expected distractor answers? "
                "Answer only 'yes' or 'no'."
            )

        if task == "next_subquestion":
            return (
                "Are the following two math subquestions semantically equivalent?\n"
                "(They ask for exactly the same quantity, even if worded differently.)\n"
                "Answer only 'yes' or 'no'.\n\n"
                f"Question 1: {gold}\n"
                f"Question 2: {prediction}"
            )

        raise ValueError(f"Unknown task: {task}")

    @torch.no_grad()
    def score(
        self,
        task: str,
        predictions: list[str],
        golds: list[str],
        batch_size: int = 1,
    ) -> tuple[list[float], list[str]]:
        scores: list[float] = []
        raw_texts: list[str] = []

        judge_prompts = [
            self.build_prompt(task=task, prediction=p, gold=g)
            for p, g in zip(predictions, golds)
        ]

        for i in range(0, len(judge_prompts), batch_size):
            batch = judge_prompts[i:i + batch_size]

            formatted = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch
            ]

            enc = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
            ).to(self.model.device)

            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            input_len = enc["input_ids"].shape[1]

            for seq in out:
                judge_text = self.tokenizer.decode(
                    seq[input_len:],
                    skip_special_tokens=True,
                ).strip()

                raw_texts.append(judge_text)
                scores.append(float(parse_yes_no(judge_text)))

        return scores, raw_texts


def load_policy_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


def load_policy_model(args, tokenizer):
    quant_config = make_quant_config(args.load_in_4bit)

    print(f"Loading policy model: {args.model_name}", flush=True)
    print(f"Policy load_in_4bit: {args.load_in_4bit}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=None if args.load_in_4bit else torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False

    return model


def build_lora_config(args):
    if not args.use_lora:
        return None

    if LoraConfig is None:
        raise ImportError(
            "peft is not available, but --use-lora was set. "
            "Install peft or run with --no-use-lora."
        )

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
    )


def build_dataset(args, task: str, tokenizer) -> Dataset:
    csv_path = args.train_csv or TASK_DEFAULT_CSV[task]
    target_col = args.target_col or TASK_TARGET_COL[task]

    df = pd.read_csv(csv_path)

    required = {args.prompt_col, target_col}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available columns: {list(df.columns)}"
        )

    if args.split_col and args.split_col in df.columns:
        df = df[
            df[args.split_col].astype(str).str.lower() == args.train_split.lower()
        ].reset_index(drop=True)

    valid = df[args.prompt_col].notna() & df[target_col].notna()
    if int((~valid).sum()):
        print(f"WARNING: dropping {int((~valid).sum())} rows with NaN prompt/target", flush=True)

    df = df[valid].reset_index(drop=True)

    if args.max_rows is not None:
        df = df.iloc[:args.max_rows].reset_index(drop=True)

    raw_prompts = [
        wrap_task_prompt(task, p)
        for p in df[args.prompt_col].astype(str).tolist()
    ]

    policy_prompts = [
        apply_policy_chat_template(
            tokenizer=tokenizer,
            user_prompt=p,
            use_chat_template=not args.no_chat_template,
        )
        for p in raw_prompts
    ]

    out = pd.DataFrame()
    out["prompt"] = policy_prompts
    out["target"] = df[target_col].astype(str)

    # Keep raw prompt for debug only.
    out["raw_prompt"] = raw_prompts

    for col in ["question", "reasoning_trace", "tree", "example_idx", "pair_index", args.split_col]:
        if col and col in df.columns and col not in out.columns:
            out[col] = df[col].astype(str)

    print("=" * 80, flush=True)
    print(f"Loaded task dataset: {task}", flush=True)
    print(f"  csv_path: {csv_path}", flush=True)
    print(f"  target_col: {target_col}", flush=True)
    print(f"  train_split: {args.train_split}", flush=True)
    print(f"  rows: {len(out)}", flush=True)
    print(f"  use_chat_template: {not args.no_chat_template}", flush=True)
    print(f"  columns: {list(out.columns)}", flush=True)

    if len(out):
        print("\nExample raw prompt:", flush=True)
        print(out.iloc[0]["raw_prompt"], flush=True)
        print("\nExample policy prompt:", flush=True)
        print(out.iloc[0]["prompt"][:2000], flush=True)
        print("\nExample target:", flush=True)
        print(out.iloc[0]["target"], flush=True)

    print("=" * 80, flush=True)

    return Dataset.from_pandas(out, preserve_index=False)


def build_grpo_config(args, task: str) -> GRPOConfig:
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    max_completion_length = (
        args.max_completion_length
        if args.max_completion_length is not None
        else TASK_MAX_COMPLETION_LENGTH[task]
    )

    desired_kwargs = {
        "output_dir": args.out_dir,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": max_completion_length,
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

    print("\nGRPOConfig accepted keys:", sorted(filtered_kwargs.keys()), flush=True)
    if dropped:
        print("GRPOConfig dropped unsupported keys:", sorted(dropped.keys()), flush=True)

    config = GRPOConfig(**filtered_kwargs)

    # Some TRL versions accept these only after construction.
    post_init_fields = {
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
    }

    for field, value in post_init_fields.items():
        if hasattr(config, field):
            setattr(config, field, value)
            print(f"Set GRPOConfig.{field} = {value}", flush=True)

    print(f"Final intended max_completion_length = {max_completion_length}", flush=True)
    print(
        f"GRPOConfig.max_completion_length after construction = "
        f"{getattr(config, 'max_completion_length', None)}",
        flush=True,
    )

    return config


def parse_args(task: str):
    parser = argparse.ArgumentParser(
        description=f"Basic TRL GRPO with Qwen judge reward for task={task}"
    )

    parser.add_argument("--model-name", default=DEFAULT_POLICY_MODEL)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--prompt-col", default="prompt")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--max-rows", type=int, default=None)

    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=None)

    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)

    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument("--judge-model-name", default=JUDGE_MODEL_NAME)
    parser.add_argument("--judge-load-in-4bit", action="store_true", default=True)
    parser.add_argument("--judge-no-load-in-4bit", dest="judge_load_in_4bit", action="store_false")
    parser.add_argument("--judge-batch-size", type=int, default=1)
    parser.add_argument("--judge-max-input-length", type=int, default=None)
    parser.add_argument("--judge-max-new-tokens", type=int, default=4)

    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")

    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-use-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--debug-print-rewards", type=int, default=10)

    args = parser.parse_args()

    if args.per_device_train_batch_size % args.num_generations != 0:
        raise ValueError(
            "per-device-train-batch-size must be divisible by num-generations. "
            f"Got batch={args.per_device_train_batch_size}, generations={args.num_generations}."
        )

    if args.judge_max_input_length is None:
        args.judge_max_input_length = 512 if task == "next_subquestion" else 1024

    return args


def run_task(task: str):
    args = parse_args(task)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = load_policy_tokenizer(args)
    dataset = build_dataset(args, task, tokenizer)
    model = load_policy_model(args, tokenizer)

    peft_config = build_lora_config(args)

    print("=" * 80, flush=True)
    print(f"TRAIN CONFIG task={task}", flush=True)
    print(f"  policy model:        {args.model_name}", flush=True)
    print(f"  judge model:         {args.judge_model_name}", flush=True)
    print(f"  policy 4bit:         {args.load_in_4bit}", flush=True)
    print(f"  judge 4bit:          {args.judge_load_in_4bit}", flush=True)
    print(f"  use_lora:            {args.use_lora}", flush=True)
    print(f"  chat_template:       {not args.no_chat_template}", flush=True)
    print(f"  num_generations:     {args.num_generations}", flush=True)
    print("=" * 80, flush=True)

    judge = QwenYesNoJudge(
        model_name=args.judge_model_name,
        load_in_4bit=args.judge_load_in_4bit,
        max_input_length=args.judge_max_input_length,
        max_new_tokens=args.judge_max_new_tokens,
    )

    debug_counter = {"n": 0}

    def qwen_judge_reward(completions, target=None, raw_prompt=None, prompt=None, **kwargs):
        if target is None:
            raise ValueError(
                "Reward function expected dataset column `target`. "
                "Check build_dataset() and remove_unused_columns=False."
            )

        predictions = [completion_to_text(c) for c in completions]
        golds = [str(x) for x in target]

        scores, judge_texts = judge.score(
            task=task,
            predictions=predictions,
            golds=golds,
            batch_size=args.judge_batch_size,
        )

        if args.debug_print_rewards > 0 and debug_counter["n"] < args.debug_print_rewards:
            raw_prompts = raw_prompt if raw_prompt is not None else [None] * len(predictions)

            for pred, gold, score, judge_text, rp in zip(
                predictions,
                golds,
                scores,
                judge_texts,
                raw_prompts,
            ):
                if debug_counter["n"] >= args.debug_print_rewards:
                    break

                print("=" * 80, flush=True)
                print(f"TASK: {task}", flush=True)
                print("RAW_PROMPT:", flush=True)
                print(rp, flush=True)
                print("COMPLETION:", flush=True)
                print(pred, flush=True)
                print("GOLD:", flush=True)
                print(gold, flush=True)
                print("JUDGE_RAW_OUTPUT:", judge_text, flush=True)
                print("JUDGE_SCORE:", score, flush=True)
                print("=" * 80, flush=True)

                debug_counter["n"] += 1

        return scores

    training_args = build_grpo_config(args, task)

    trainer_kwargs = {
        "model": model,
        "reward_funcs": [qwen_judge_reward],
        "args": training_args,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }

    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    try:
        trainer = GRPOTrainer(**trainer_kwargs)
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise

        print(
            "GRPOTrainer does not accept processing_class; falling back to tokenizer=tokenizer",
            flush=True,
        )

        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        trainer = GRPOTrainer(**trainer_kwargs)

    print(
        f"trainer.args.max_completion_length actual = "
        f"{getattr(trainer.args, 'max_completion_length', None)}",
        flush=True,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print(f"Saved TRL GRPO Qwen-judge model for task={task} to {args.out_dir}", flush=True)
