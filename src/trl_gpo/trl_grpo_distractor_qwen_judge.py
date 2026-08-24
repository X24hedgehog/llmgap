#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import re
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------
# Compatibility patch:
# Some TRL versions expect torch.distributed.fsdp.FSDPModule, but some
# PyTorch builds do not expose it. We are not using FSDP here, so this is
# a harmless fallback before importing TRL.
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


DEFAULT_POLICY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))

        if completion and isinstance(completion[-1], list):
            return completion_to_text(completion[-1])

    return str(completion)


def make_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def parse_gold_distractors(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass

    return [text]


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("$", "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def build_dataset(args: argparse.Namespace) -> Dataset:
    df = pd.read_csv(args.train_csv)

    required = {
        args.prompt_col,
        args.question_col,
        args.gold_distractor_col,
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available columns: {list(df.columns)}"
        )

    if args.split_col and args.split_col in df.columns:
        df = df[
            df[args.split_col].astype(str).str.lower() == args.train_split.lower()
        ].reset_index(drop=True)

    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].reset_index(drop=True)

    df = df.dropna(
        subset=[
            args.prompt_col,
            args.question_col,
            args.gold_distractor_col,
        ]
    ).reset_index(drop=True)

    out = pd.DataFrame()

    out["prompt"] = df[args.prompt_col].astype(str)
    out["question"] = df[args.question_col].astype(str)
    out["gold_distractor_answers"] = df[args.gold_distractor_col].astype(str)

    if args.correct_answer_col and args.correct_answer_col in df.columns:
        out["correct_answer"] = df[args.correct_answer_col].astype(str)
    else:
        out["correct_answer"] = ""

    if args.misconception_col and args.misconception_col in df.columns:
        out["misconception_name"] = df[args.misconception_col].astype(str)
    else:
        out["misconception_name"] = ""

    print("Loaded distractor dataset", flush=True)
    print(f"  path: {args.train_csv}", flush=True)
    print(f"  rows: {len(out)}", flush=True)
    print(f"  columns used: {list(out.columns)}", flush=True)

    if len(out) > 0:
        print("\nExample policy prompt:", flush=True)
        print(out.iloc[0]["prompt"], flush=True)
        print("\nExample gold distractor answers:", flush=True)
        print(out.iloc[0]["gold_distractor_answers"], flush=True)

    return Dataset.from_pandas(out, preserve_index=False)


def build_judge_prompt(
    problem_prompt: str,
    question: str,
    gold_distractors: list[str],
    candidate_answer: str,
    correct_answer: str = "",
    misconception_name: str = "",
    include_policy_prompt: bool = True,
) -> str:
    candidate_answer = str(candidate_answer).strip()
    gold_list = "\n".join([f"- {g}" for g in gold_distractors]) if gold_distractors else "- (none)"

    context = ""
    if include_policy_prompt:
        context += f"\n\nThe policy model was given this prompt:\n{problem_prompt}"

    if str(correct_answer).strip():
        context += f"\n\nCorrect answer (must NOT be produced as distractor):\n{correct_answer}"

    if str(misconception_name).strip():
        context += f"\n\nTarget misconception context:\n{misconception_name}"

    return f"""You are judging a generated distractor answer for a math problem.

Original math problem:
{question}{context}

Reference acceptable distractor answers:
{gold_list}

Candidate generated distractor answer:
{candidate_answer}

Decide whether the candidate is an acceptable distractor.

Answer \"yes\" only if:
- it is semantically equivalent to at least one reference distractor answer
- and it is not equivalent to the correct answer
- and it is a concise distractor answer (not a full long derivation)

Answer \"no\" otherwise.

Answer only yes or no."""


class QwenYesNoJudge:
    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool,
        max_input_length: int,
        max_new_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens

        print(f"Loading Qwen judge model: {model_name}", flush=True)

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
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.no_grad()
    def judge_yes_no(
        self,
        judge_prompts: list[str],
        batch_size: int,
    ) -> list[int]:
        scores: list[int] = []

        for start in range(0, len(judge_prompts), batch_size):
            batch = judge_prompts[start : start + batch_size]

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

                verdicts = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
                final_verdict = verdicts[-1].lower() if verdicts else "no"
                scores.append(1 if final_verdict == "yes" else 0)

        return scores


def load_policy_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    quant_config = make_quant_config(args.load_in_4bit)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and not args.load_in_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.sft_adapter_path:
        print(f"Loading SFT adapter for RL initialization: {args.sft_adapter_path}", flush=True)
        model = PeftModel.from_pretrained(
            model,
            args.sft_adapter_path,
            is_trainable=True,
        )
        model.print_trainable_parameters()

    return model, tokenizer


def build_grpo_config(args: argparse.Namespace) -> GRPOConfig:
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

    filtered_kwargs = {k: v for k, v in desired_kwargs.items() if k in accepted}
    dropped = {k: v for k, v in desired_kwargs.items() if k not in accepted}

    print("\nGRPOConfig accepted keys:", sorted(filtered_kwargs.keys()), flush=True)
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
    if args.sft_adapter_path:
        return None

    if not args.use_peft:
        return None

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )


def one_line_answer_format_score(text: str) -> float:
    s = str(text).strip()
    if not s:
        return 0.0
    if "\n" in s:
        return 0.0
    return 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRL GRPO for EEDI distractor task with Qwen 7B yes/no judge reward."
    )

    parser.add_argument("--model-name", default=DEFAULT_POLICY_MODEL)
    parser.add_argument(
        "--sft-adapter-path",
        default=None,
        help="Optional path to SFT LoRA adapter checkpoint. If provided, RL starts from this adapter.",
    )

    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--prompt-col", default="prompt")
    parser.add_argument("--question-col", default="question")
    parser.add_argument("--gold-distractor-col", default="target_distractor_answers")
    parser.add_argument("--correct-answer-col", default="correct_answer")
    parser.add_argument("--misconception-col", default="misconception_name")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--max-rows", type=int, default=None)

    parser.add_argument("--judge-model-name", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-load-in-4bit", action="store_true")
    parser.add_argument("--judge-batch-size", type=int, default=1)
    parser.add_argument("--judge-max-input-length", type=int, default=2048)
    parser.add_argument("--judge-max-new-tokens", type=int, default=8)
    parser.add_argument("--include-policy-prompt-in-judge", action="store_true")

    parser.add_argument("--format-reward-weight", type=float, default=0.0)
    parser.add_argument("--judge-reward-scale", type=float, default=1.0)

    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=64)

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)

    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument("--use-peft", action="store_true")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--debug-print-rewards", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = build_dataset(args)
    model, tokenizer = load_policy_and_tokenizer(args)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    judge = QwenYesNoJudge(
        model_name=args.judge_model_name,
        load_in_4bit=args.judge_load_in_4bit,
        max_input_length=args.judge_max_input_length,
        max_new_tokens=args.judge_max_new_tokens,
    )

    debug_counter = {"n": 0}

    def qwen_judge_reward(
        completions,
        prompt=None,
        question=None,
        gold_distractor_answers=None,
        correct_answer=None,
        misconception_name=None,
        **kwargs,
    ):
        if question is None or gold_distractor_answers is None:
            raise ValueError(
                "Reward function expected dataset columns `question` and "
                "`gold_distractor_answers`. Check build_dataset() and "
                "remove_unused_columns=False."
            )

        if prompt is None:
            prompt = [""] * len(completions)
        if correct_answer is None:
            correct_answer = [""] * len(completions)
        if misconception_name is None:
            misconception_name = [""] * len(completions)

        judge_prompts = []
        candidates = []
        parsed_gold_lists = []

        for completion, p, q, golds, correct, miscon in zip(
            completions,
            prompt,
            question,
            gold_distractor_answers,
            correct_answer,
            misconception_name,
        ):
            candidate = completion_to_text(completion).strip()
            candidates.append(candidate)

            gold_list = parse_gold_distractors(golds)
            parsed_gold_lists.append(gold_list)

            judge_prompts.append(
                build_judge_prompt(
                    problem_prompt=str(p),
                    question=str(q),
                    gold_distractors=gold_list,
                    candidate_answer=candidate,
                    correct_answer=str(correct),
                    misconception_name=str(miscon),
                    include_policy_prompt=args.include_policy_prompt_in_judge,
                )
            )

        yes_no_scores = judge.judge_yes_no(judge_prompts, batch_size=args.judge_batch_size)

        rewards = [args.judge_reward_scale * float(score) for score in yes_no_scores]

        if args.format_reward_weight > 0:
            rewards = [
                r + args.format_reward_weight * one_line_answer_format_score(c)
                for r, c in zip(rewards, candidates)
            ]

        if args.debug_print_rewards > 0:
            for cand, gold_list, score, reward in zip(
                candidates,
                parsed_gold_lists,
                yes_no_scores,
                rewards,
            ):
                if debug_counter["n"] >= args.debug_print_rewards:
                    break

                print("=" * 100, flush=True)
                print("CANDIDATE DISTRACTOR:", flush=True)
                print(cand, flush=True)
                print("GOLD DISTRACTOR ANSWERS:", flush=True)
                for item in gold_list:
                    print(f"- {item}", flush=True)
                print(f"QWEN JUDGE YES/NO SCORE: {score}", flush=True)
                print(f"FINAL REWARD: {reward:.4f}", flush=True)
                print("=" * 100, flush=True)

                debug_counter["n"] += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return rewards

    peft_config = build_peft_config(args)
    training_args = build_grpo_config(args)

    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=qwen_judge_reward,
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
            reward_funcs=qwen_judge_reward,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print(f"Saved TRL GRPO distractor model to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
