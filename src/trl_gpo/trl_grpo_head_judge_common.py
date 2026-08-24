#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import inspect
import os
import re
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------
# Compatibility patch for some TRL / torch FSDP combinations
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


# ---------------------------------------------------------------------
# Task conventions aligned with reasoning-efficiency/experiments/proof_search/run_inference.py
# ---------------------------------------------------------------------

JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

CORRECT_ANSWER_INSTRUCTION = (
    "\n\nClearly solve the problem step by step.\n"
    "End with exactly one final line:\n"
    "#### <number>\n"
    "After that line, immediately stop."
)

DISTRACTOR_INSTRUCTION = (
    "\n\nIncorrect Answer: Let's think step by step."
)

TASK_TARGET_COL = {
    "correct_answer": "target_answer",
    "next_subquestion": "next_subquestion",
    "distractor": "target_distractor_answers",
}

TASK_MAX_COMPLETION_LENGTH = {
    "correct_answer": 96,
    "next_subquestion": 48,
    "distractor": 256,
}

TASK_DEFAULT_CSV = {
    "correct_answer": "/cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs_gsm8k.csv",
    "next_subquestion": "/cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs_gsm8k.csv",
    "distractor": "/cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/distractor_pairs_gsm8k.csv",
}


# ---------------------------------------------------------------------
# Prompt wrapping
# ---------------------------------------------------------------------

def wrap_correct_answer_prompt(prompt: str) -> str:
    return str(prompt).rstrip() + CORRECT_ANSWER_INSTRUCTION


def wrap_distractor_prompt(prompt: str) -> str:
    return str(prompt).rstrip() + DISTRACTOR_INSTRUCTION


def apply_task_prompt(task: str, prompt: str) -> str:
    if task == "correct_answer":
        return wrap_correct_answer_prompt(prompt)
    if task == "distractor":
        return wrap_distractor_prompt(prompt)
    if task == "next_subquestion":
        return str(prompt)
    raise ValueError(f"Unknown task: {task}")


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

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
    text = str(text).strip().lower()
    matches = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if not matches:
        return 0
    return 1 if matches[-1].lower() == "yes" else 0


def make_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


# ---------------------------------------------------------------------
# Correct-answer extraction and reward analysis
# ---------------------------------------------------------------------

FINAL_LINE_RE = re.compile(
    r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)"
)

STRICT_FINAL_LINE_AT_END_RE = re.compile(
    r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*$"
)


def extract_final_answer_number(text: str) -> str | None:
    """
    Extract a final numeric answer from GSM8K-style completion.

    Order:
    1. First #### <number> marker
    2. \\boxed{number}
    3. "final answer is/:" pattern
    4. fallback last number anywhere
    """
    text = str(text)

    match = FINAL_LINE_RE.search(text)
    if match:
        return match.group(1).replace(",", "").strip()

    boxed = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        inner = boxed.group(1).strip()
        num_match = re.search(r"-?[\d,]+(?:\.\d+)?", inner)
        if num_match:
            return num_match.group(0).replace(",", "").strip()

    final_ans = re.search(
        r"final answer(?:\s+is)?[:\s]*\$?\s*(-?[\d,]+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if final_ans:
        return final_ans.group(1).replace(",", "").strip()

    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "").strip()

    return None


def numbers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False

    gold_str = str(gold)
    gold_extracted = extract_final_answer_number(gold_str)
    gold_value = gold_extracted if gold_extracted is not None else gold_str

    try:
        return abs(float(str(pred).replace(",", "")) - float(str(gold_value).replace(",", ""))) < 1e-6
    except ValueError:
        return str(pred).strip() == str(gold_value).strip()


def analyze_correct_answer_ending(text: str, max_words: int = 120) -> dict:
    """
    Gives a detailed breakdown of formatting/stopping behavior.
    """
    text = str(text)
    stripped = text.strip()

    first_final = FINAL_LINE_RE.search(text)
    strict_final = STRICT_FINAL_LINE_AT_END_RE.search(stripped.replace(",", ""))

    has_final_line = first_final is not None
    ends_with_final_line = strict_final is not None

    trailing_text = ""
    if first_final is not None:
        trailing_text = text[first_final.end():].strip()

    extracted = extract_final_answer_number(text)
    word_count = len(text.split())

    return {
        "has_final_line": has_final_line,
        "ends_with_final_line": ends_with_final_line,
        "has_trailing_text": bool(trailing_text),
        "trailing_text": trailing_text[:300],
        "extracted_answer": extracted,
        "word_count": word_count,
        "too_long": word_count > max_words,
    }


def loose_final_marker_score(text: str) -> float:
    """
    Partial credit: did the model at least produce #### <number> somewhere?
    This is intentionally easier than strict final-format reward.
    """
    return 1.0 if FINAL_LINE_RE.search(str(text)) else 0.0


def strict_final_answer_format_score(text: str) -> float:
    """
    Full format credit: output ends cleanly with #### <number>.
    """
    clean = str(text).strip().replace(",", "")
    return 1.0 if STRICT_FINAL_LINE_AT_END_RE.search(clean) else 0.0


def correct_answer_penalty(
    text: str,
    max_words: int = 120,
    trailing_penalty: float = 0.2,
    rambling_penalty: float = 0.05,
) -> float:
    """
    Targeted penalty:
    - Stronger penalty if model writes after #### <number>
    - Smaller penalty if no final answer line and output is too long
    """
    analysis = analyze_correct_answer_ending(text, max_words=max_words)

    if analysis["has_final_line"] and analysis["has_trailing_text"]:
        return trailing_penalty

    if (not analysis["has_final_line"]) and analysis["too_long"]:
        return rambling_penalty

    return 0.0


# ---------------------------------------------------------------------
# Other task format rewards
# ---------------------------------------------------------------------

def one_question_format_score(text: str) -> float:
    text = str(text).strip()

    if not text:
        return 0.0
    if "\n" in text:
        return 0.0
    if "?" not in text:
        return 0.0
    if text.count("?") > 1:
        return 0.0
    if "<<" in text or ">>" in text:
        return 0.0
    if len(text.split()) > 35:
        return 0.0

    return 1.0


def distractor_format_score(text: str) -> float:
    text = str(text).strip()
    return 1.0 if len(text.split()) >= 5 else 0.0


# ---------------------------------------------------------------------
# Qwen yes/no judge, matching run_inference.py mechanism
# ---------------------------------------------------------------------

class QwenYesNoJudge:
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
    ) -> tuple[list[int], list[str]]:
        scores: list[int] = []
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
                text = self.tokenizer.decode(
                    seq[input_len:],
                    skip_special_tokens=True,
                ).strip()
                raw_texts.append(text)
                scores.append(parse_yes_no(text))

        return scores, raw_texts


# ---------------------------------------------------------------------
# Model loading and head-only training
# ---------------------------------------------------------------------

def configure_head_only_training(model, train_embed_too: bool = False) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    trainable_modules = []

    if hasattr(model, "lm_head") and model.lm_head is not None:
        for param in model.lm_head.parameters():
            param.requires_grad = True
        trainable_modules.append("lm_head")
    else:
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            for param in output_emb.parameters():
                param.requires_grad = True
            trainable_modules.append("output_embeddings")

    if train_embed_too:
        input_emb = model.get_input_embeddings()
        if input_emb is not None:
            for param in input_emb.parameters():
                param.requires_grad = True
            trainable_modules.append("input_embeddings")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 80, flush=True)
    print("HEAD-ONLY TRAINING ENABLED", flush=True)
    print("Trainable modules:", trainable_modules, flush=True)
    print(f"Trainable parameters: {trainable_params:,}", flush=True)
    print(f"Total parameters:     {total_params:,}", flush=True)
    print(f"Trainable fraction:   {100.0 * trainable_params / max(total_params, 1):.6f}%", flush=True)
    print("=" * 80, flush=True)


def load_policy_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.load_in_4bit:
        print(
            "WARNING: policy model is loaded in 4-bit. "
            "For head-only training, prefer no --load-in-4bit if memory allows.",
            flush=True,
        )

    quant_config = make_quant_config(args.load_in_4bit)

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

    configure_head_only_training(
        model,
        train_embed_too=args.train_embed_too,
    )

    return model, tokenizer


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def build_dataset(args: argparse.Namespace, task: str) -> Dataset:
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

    valid_mask = df[args.prompt_col].notna() & df[target_col].notna()
    dropped = int((~valid_mask).sum())
    if dropped:
        print(f"WARNING: dropping {dropped} rows with NaN prompt/target", flush=True)

    df = df[valid_mask].reset_index(drop=True)

    if args.max_rows is not None:
        df = df.iloc[:args.max_rows].reset_index(drop=True)

    out = pd.DataFrame()
    out["prompt"] = [
        apply_task_prompt(task, p)
        for p in df[args.prompt_col].astype(str).tolist()
    ]
    out["target"] = df[target_col].astype(str)

    for col in ["question", "reasoning_trace", "tree", "example_idx", "pair_index", args.split_col]:
        if col and col in df.columns and col not in out.columns:
            out[col] = df[col].astype(str)

    print("=" * 80, flush=True)
    print(f"Loaded task dataset: {task}", flush=True)
    print(f"  csv_path: {csv_path}", flush=True)
    print(f"  target_col: {target_col}", flush=True)
    print(f"  train_split: {args.train_split}", flush=True)
    print(f"  rows: {len(out)}", flush=True)
    print(f"  columns: {list(out.columns)}", flush=True)
    if len(out):
        print("\nExample prompt:", flush=True)
        print(out.iloc[0]["prompt"], flush=True)
        print("\nExample target:", flush=True)
        print(out.iloc[0]["target"], flush=True)
    print("=" * 80, flush=True)

    return Dataset.from_pandas(out, preserve_index=False)


# ---------------------------------------------------------------------
# GRPO config
# ---------------------------------------------------------------------

def build_grpo_config(args: argparse.Namespace, task: str) -> GRPOConfig:
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
        "gradient_checkpointing": False,
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

    actual_max_completion_length = getattr(config, "max_completion_length", None)

    print(f"Final intended max_completion_length = {max_completion_length}", flush=True)
    print(
        f"GRPOConfig.max_completion_length after construction = {actual_max_completion_length}",
        flush=True,
    )

    if actual_max_completion_length != max_completion_length:
        print(
            "WARNING: installed TRL version did not accept/keep max_completion_length.",
            flush=True,
        )

    return config


# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------

def parse_args(task: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Head-only TRL GRPO for task={task}."
    )

    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
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
    parser.add_argument("--lr", type=float, default=1e-5)

    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument("--judge-model-name", default=JUDGE_MODEL_NAME)
    parser.add_argument("--judge-load-in-4bit", action="store_true")
    parser.add_argument("--judge-batch-size", type=int, default=1)
    parser.add_argument("--judge-max-input-length", type=int, default=None)
    parser.add_argument("--judge-max-new-tokens", type=int, default=4)

    # Correct-answer reward components
    parser.add_argument("--exact-reward-weight", type=float, default=1.0)
    parser.add_argument("--loose-marker-reward-weight", type=float, default=0.1)
    parser.add_argument("--strict-format-reward-weight", type=float, default=0.2)
    parser.add_argument("--trailing-penalty-weight", type=float, default=0.2)
    parser.add_argument("--rambling-penalty-weight", type=float, default=0.05)
    parser.add_argument("--max-words-before-rambling", type=int, default=120)

    # Other task rewards
    parser.add_argument("--format-reward-weight", type=float, default=None)
    parser.add_argument("--judge-reward-weight", type=float, default=None)
    parser.add_argument("--similarity-reward-weight", type=float, default=0.0)

    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--train-embed-too", action="store_true")

    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--debug-print-rewards", type=int, default=10)

    args = parser.parse_args()

    if args.format_reward_weight is None:
        args.format_reward_weight = {
            "correct_answer": 0.0,
            "next_subquestion": 0.2,
            "distractor": 0.0,
        }[task]

    if args.judge_reward_weight is None:
        args.judge_reward_weight = 0.0 if task == "correct_answer" else 1.0

    if args.per_device_train_batch_size % args.num_generations != 0:
        raise ValueError(
            "per-device-train-batch-size must be divisible by num-generations. "
            f"Got batch={args.per_device_train_batch_size}, generations={args.num_generations}."
        )

    if args.judge_max_input_length is None:
        args.judge_max_input_length = 512 if task == "next_subquestion" else 1024

    return args


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def run_task(task: str) -> None:
    args = parse_args(task)
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = build_dataset(args, task)
    model, tokenizer = load_policy_and_tokenizer(args)

    need_judge = (task != "correct_answer") or (args.judge_reward_weight > 0)

    judge = None
    if need_judge:
        judge = QwenYesNoJudge(
            model_name=args.judge_model_name,
            load_in_4bit=args.judge_load_in_4bit,
            max_input_length=args.judge_max_input_length,
            max_new_tokens=args.judge_max_new_tokens,
        )
    else:
        print(
            "Skipping Qwen judge load for correct_answer because judge_reward_weight=0.",
            flush=True,
        )

    print("=" * 80, flush=True)
    print(f"REWARD CONFIG task={task}", flush=True)
    print(f"  exact_reward_weight          = {args.exact_reward_weight}", flush=True)
    print(f"  loose_marker_reward_weight   = {args.loose_marker_reward_weight}", flush=True)
    print(f"  strict_format_reward_weight  = {args.strict_format_reward_weight}", flush=True)
    print(f"  trailing_penalty_weight      = {args.trailing_penalty_weight}", flush=True)
    print(f"  rambling_penalty_weight      = {args.rambling_penalty_weight}", flush=True)
    print(f"  format_reward_weight         = {args.format_reward_weight}", flush=True)
    print(f"  judge_reward_weight          = {args.judge_reward_weight}", flush=True)
    print(f"  similarity_reward_weight     = {args.similarity_reward_weight}", flush=True)
    print(f"  judge_loaded                 = {need_judge}", flush=True)
    print("=" * 80, flush=True)

    debug_counter = {"n": 0}

    def task_reward_fn(completions, target=None, prompt=None, prompts=None, **kwargs):
        if target is None:
            raise ValueError(
                "Reward function expected dataset column `target`. "
                "Check build_dataset() and remove_unused_columns=False."
            )

        if prompts is None:
            prompts = prompt
        if prompts is None:
            prompts = kwargs.get("prompts")
        if prompts is None:
            prompts = kwargs.get("prompt")

        predictions = [completion_to_text(c) for c in completions]
        golds = [str(x) for x in target]
        prompt_texts = (
            [str(p) for p in prompts]
            if prompts is not None
            else [None] * len(predictions)
        )

        judge_scores: list[int] | None = None
        judge_texts: list[str] | None = None

        if judge is not None and args.judge_reward_weight > 0:
            judge_scores, judge_texts = judge.score(
                task=task,
                predictions=predictions,
                golds=golds,
                batch_size=args.judge_batch_size,
            )

        rewards = []

        for idx, (prompt_text, pred, gold) in enumerate(zip(prompt_texts, predictions, golds)):
            exact = 0.0
            loose_marker = 0.0
            strict_format = 0.0
            penalty = 0.0
            extracted_answer = None
            ending = {}

            if task == "correct_answer":
                ending = analyze_correct_answer_ending(
                    pred,
                    max_words=args.max_words_before_rambling,
                )
                extracted_answer = ending["extracted_answer"]
                exact = 1.0 if numbers_match(extracted_answer, gold) else 0.0
                loose_marker = 1.0 if ending["has_final_line"] else 0.0
                strict_format = 1.0 if ending["ends_with_final_line"] else 0.0

                if ending["has_final_line"] and ending["has_trailing_text"]:
                    penalty = args.trailing_penalty_weight
                elif (not ending["has_final_line"]) and ending["too_long"]:
                    penalty = args.rambling_penalty_weight

                reward = (
                    args.exact_reward_weight * exact
                    + args.loose_marker_reward_weight * loose_marker
                    + args.strict_format_reward_weight * strict_format
                    - penalty
                )

            elif task == "next_subquestion":
                fmt = one_question_format_score(pred)

                sim = 0.0
                if args.similarity_reward_weight > 0:
                    sim = difflib.SequenceMatcher(
                        None,
                        pred.strip().lower(),
                        gold.strip().lower(),
                    ).ratio()

                judge_score = float(judge_scores[idx]) if judge_scores is not None else 0.0

                reward = (
                    args.judge_reward_weight * judge_score
                    + args.format_reward_weight * fmt
                    + args.similarity_reward_weight * sim
                )

            elif task == "distractor":
                fmt = distractor_format_score(pred)
                judge_score = float(judge_scores[idx]) if judge_scores is not None else 0.0

                reward = (
                    args.judge_reward_weight * judge_score
                    + args.format_reward_weight * fmt
                )

            else:
                raise ValueError(f"Unknown task: {task}")

            rewards.append(float(reward))

            if args.debug_print_rewards > 0 and debug_counter["n"] < args.debug_print_rewards:
                print("=" * 80, flush=True)
                print(f"TASK: {task}", flush=True)
                print("PROMPT:", flush=True)
                print(prompt_text, flush=True)
                print("COMPLETION:", flush=True)
                print(pred, flush=True)
                print("GOLD:", flush=True)
                print(gold, flush=True)

                if task == "correct_answer":
                    print("EXTRACTED_ANSWER:", extracted_answer, flush=True)
                    print("EXACT_REWARD:", exact, flush=True)
                    print("LOOSE_MARKER_REWARD:", loose_marker, flush=True)
                    print("STRICT_FORMAT_REWARD:", strict_format, flush=True)
                    print("HAS_FINAL_LINE:", ending.get("has_final_line"), flush=True)
                    print("ENDS_WITH_FINAL_LINE:", ending.get("ends_with_final_line"), flush=True)
                    print("HAS_TRAILING_TEXT:", ending.get("has_trailing_text"), flush=True)
                    print("WORD_COUNT:", ending.get("word_count"), flush=True)
                    print("TOO_LONG:", ending.get("too_long"), flush=True)
                    print("PENALTY:", penalty, flush=True)

                if judge_scores is not None:
                    print("JUDGE_PROMPT:", flush=True)
                    print(judge.build_prompt(task=task, prediction=pred, gold=gold), flush=True)
                    print("JUDGE_RAW_OUTPUT:", judge_texts[idx], flush=True)
                    print("JUDGE_SCORE:", judge_scores[idx], flush=True)

                if task == "next_subquestion":
                    print("QUESTION_FORMAT_SCORE:", one_question_format_score(pred), flush=True)
                    if args.similarity_reward_weight > 0:
                        print(
                            "SIMILARITY_SCORE:",
                            difflib.SequenceMatcher(
                                None,
                                pred.strip().lower(),
                                gold.strip().lower(),
                            ).ratio(),
                            flush=True,
                        )

                print("TOTAL_REWARD:", reward, flush=True)
                print("=" * 80, flush=True)
                debug_counter["n"] += 1

        return rewards

    training_args = build_grpo_config(args, task)

    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[task_reward_fn],
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
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
            reward_funcs=[task_reward_fn],
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

    print(
        f"trainer.args.max_completion_length (actual) = "
        f"{getattr(trainer.args, 'max_completion_length', None)}",
        flush=True,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print(f"Saved head-only TRL GRPO model for task={task} to {args.out_dir}", flush=True)
