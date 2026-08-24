from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# ============================================================
# Prompt templates
# ============================================================

GSM8K_STEP_BY_STEP_PROMPT = """Solve the following grade school math problem step by step.

Rules:
- Be concise.
- The last line must be exactly: #### <number>
- Do not write anything after the final answer line.

Question: {question}
Answer:"""

GSM8K_DIRECT_PROMPT = """Solve the following grade school math problem.
Write the final answer in the format: #### <number>

Question: {question}
Answer:"""

PROMPT_TEMPLATES = {
    "step_by_step": GSM8K_STEP_BY_STEP_PROMPT,
    "direct": GSM8K_DIRECT_PROMPT,
}

DEFAULT_PROMPT_STYLE = "step_by_step"


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    model_name: str
    train_csv: str
    out_dir: str

    prompt_style: str = DEFAULT_PROMPT_STYLE

    seed: int = 42
    max_rows: int = 200

    num_samples: int = 4
    batch_size: int = 2

    max_prompt_length: int = 512
    max_new_tokens: int = 256

    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True

    lr: float = 1e-7
    epochs: int = 1
    max_grad_norm: float = 0.5

    device: str = "cuda"

    # Rule-based reward
    format_reward: float = 0.1

    # GRPO clipping
    clip_eps: float = 0.2

    # Auxiliary SFT on golden reasoning trace
    sft_coef: float = 0.05
    max_sft_length: int = 768

    # Debugging
    debug: bool = False
    debug_print_samples: int = 0


# ============================================================
# Debug helper
# ============================================================

def print_debug(cfg: TrainConfig, *args, **kwargs) -> None:
    if cfg.debug:
        print(*args, **kwargs, flush=True)


# ============================================================
# Data / prompt
# ============================================================

def load_gsm8k_dataframe(csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"question", "target_answer", "reasoning_trace"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "split" in df.columns:
        train_df = df[df["split"].astype(str).str.lower() == "train"]
        if len(train_df) > 0:
            df = train_df

    df = df.reset_index(drop=True)

    if max_rows is not None:
        df = df.iloc[:max_rows].reset_index(drop=True)

    return df


def build_prompt(question: str, prompt_style: str) -> str:
    template = PROMPT_TEMPLATES[prompt_style]
    return template.format(question=question)


# ============================================================
# Reward functions
# ============================================================

def extract_last_number(text: str) -> str | None:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not numbers:
        return None
    return numbers[-1]


def extract_final_answer(response: str) -> str:
    response = str(response).replace(",", "")

    # Prefer explicit GSM8K final-answer marker.
    # Allows: #### 10, #### $10, #### -3.5
    marker_match = re.search(
        r"####\s*\$?\s*(-?\d+(?:\.\d+)?)",
        response,
    )
    if marker_match:
        return marker_match.group(1)

    # Then try LaTeX boxed answer, e.g. \boxed{72}
    boxed_match = re.search(
        r"\\boxed\{\s*\$?\s*(-?\d+(?:\.\d+)?)\s*\}",
        response,
    )
    if boxed_match:
        return boxed_match.group(1)

    # Then try phrases like "final answer is 72"
    final_match = re.search(
        r"final answer (?:is|:)\s*\$?\s*(-?\d+(?:\.\d+)?)",
        response,
        flags=re.IGNORECASE,
    )
    if final_match:
        return final_match.group(1)

    last_number = extract_last_number(response)
    if last_number is not None:
        return last_number

    return response.strip()


def normalize_answer(answer: str) -> str:
    answer = str(answer).strip().replace(",", "")

    marker_match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer)
    if marker_match:
        answer = marker_match.group(1)

    number = extract_last_number(answer)
    if number is not None:
        answer = number

    try:
        value = float(answer)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return answer.strip()


def answers_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def compute_correct_answer_reward(response: str, gold_answer: str) -> float:
    pred = extract_final_answer(response)
    return 1.0 if answers_match(pred, gold_answer) else 0.0


def compute_format_reward(response: str) -> float:
    response = str(response).replace(",", "")

    return 1.0 if re.search(
        r"####\s*\$?\s*-?\d+(?:\.\d+)?",
        response,
    ) else 0.0


def compute_total_reward(
    response: str,
    gold_answer: str,
    cfg: TrainConfig,
) -> dict[str, float]:
    answer_correct = compute_correct_answer_reward(response, gold_answer)
    format_ok = compute_format_reward(response)

    reward = answer_correct + cfg.format_reward * format_ok

    return {
        "reward": float(reward),
        "answer_correct": float(answer_correct),
        "format_reward": float(format_ok),
    }


# ============================================================
# Model loading
# ============================================================

def load_policy_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    model.to(device)
    model.train()

    return model, tokenizer


def debug_model_after_loading(model, tokenizer, cfg: TrainConfig) -> None:
    print_debug(cfg, "model name:", cfg.model_name)
    print_debug(cfg, "model class:", model.__class__)
    print_debug(cfg, "tokenizer vocab size:", len(tokenizer))
    print_debug(cfg, "model vocab size:", model.config.vocab_size)
    print_debug(cfg, "embedding size:", model.get_input_embeddings().weight.shape[0])

    first_param = next(model.parameters())
    print_debug(cfg, "first param dtype:", first_param.dtype)
    print_debug(cfg, "first param finite:", torch.isfinite(first_param).all().item())

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        raise RuntimeError(
            f"Tokenizer has more tokens than model embeddings: "
            f"len(tokenizer)={len(tokenizer)}, "
            f"emb_size={model.get_input_embeddings().weight.shape[0]}"
        )


# ============================================================
# Generation / rollout
# ============================================================

@torch.no_grad()
def sample_one_response(
    model,
    tokenizer,
    prompt: str,
    cfg: TrainConfig,
) -> str:
    model.eval()

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_prompt_length,
    ).to(model.device)

    max_input_id = enc["input_ids"].max().item()
    emb_size = model.get_input_embeddings().weight.shape[0]
    if max_input_id >= emb_size:
        raise RuntimeError(
            f"Input token id exceeds embedding size: "
            f"max_input_id={max_input_id}, emb_size={emb_size}"
        )

    gen_kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "remove_invalid_values": True,
    }

    if cfg.do_sample:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": max(float(cfg.temperature), 1e-5),
                "top_p": float(cfg.top_p),
                "renormalize_logits": True,
            }
        )
    else:
        gen_kwargs.update({"do_sample": False})

    generation = model.generate(
        **enc,
        **gen_kwargs,
    )

    response_ids = generation[0, enc["input_ids"].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    model.train()
    return response


def sample_rollouts_for_batch(
    model,
    tokenizer,
    prompts: list[str],
    gold_answers: list[str],
    cfg: TrainConfig,
) -> list[dict[str, Any]]:
    rollouts = []
    printed = 0

    for group_id, (prompt, gold_answer) in enumerate(zip(prompts, gold_answers)):
        for sample_id in range(cfg.num_samples):
            response = sample_one_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                cfg=cfg,
            )

            reward_info = compute_total_reward(
                response=response,
                gold_answer=gold_answer,
                cfg=cfg,
            )

            rollout = {
                "group_id": group_id,
                "sample_id": sample_id,
                "prompt": prompt,
                "response": response,
                "gold_answer": gold_answer,
                **reward_info,
            }

            rollouts.append(rollout)

            if cfg.debug_print_samples > 0 and printed < cfg.debug_print_samples:
                print("=" * 80, flush=True)
                print("PROMPT:", flush=True)
                print(prompt, flush=True)
                print("RESPONSE:", flush=True)
                print(response, flush=True)
                print("GOLD:", gold_answer, flush=True)
                print("EXTRACTED:", extract_final_answer(response), flush=True)
                print("REWARD_INFO:", reward_info, flush=True)
                print("=" * 80, flush=True)
                printed += 1

    return rollouts


# ============================================================
# GRPO utilities
# ============================================================

def build_logprob_batch(
    rollouts: list[dict[str, Any]],
    tokenizer,
    device,
) -> dict[str, torch.Tensor]:
    input_ids_list = []
    response_mask_list = []

    for rollout in rollouts:
        prompt = rollout["prompt"]
        response = rollout["response"]

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        response_ids = tokenizer(
            response,
            add_special_tokens=False,
        )["input_ids"]

        if len(response_ids) == 0:
            response_ids = [tokenizer.eos_token_id]

        full_ids = prompt_ids + response_ids

        response_mask = [False] * len(prompt_ids) + [True] * len(response_ids)

        input_ids_list.append(full_ids)
        response_mask_list.append(response_mask)

    max_len = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id

    padded_input_ids = []
    padded_attention_mask = []
    padded_response_mask = []

    for ids, mask in zip(input_ids_list, response_mask_list):
        pad_len = max_len - len(ids)

        padded_input_ids.append(ids + [pad_id] * pad_len)
        padded_attention_mask.append([1] * len(ids) + [0] * pad_len)
        padded_response_mask.append(mask + [False] * pad_len)

    return {
        "input_ids": torch.tensor(
            padded_input_ids,
            dtype=torch.long,
            device=device,
        ),
        "attention_mask": torch.tensor(
            padded_attention_mask,
            dtype=torch.long,
            device=device,
        ),
        "response_mask": torch.tensor(
            padded_response_mask,
            dtype=torch.bool,
            device=device,
        ),
    }


def compute_token_logprobs(
    model,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits

    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite logits in compute_token_logprobs")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    shifted_response_mask = response_mask[:, 1:]

    vocab_size = shifted_logits.shape[-1]
    if shifted_labels.max().item() >= vocab_size:
        raise RuntimeError(
            f"Label id exceeds vocab size: "
            f"max_label={shifted_labels.max().item()}, vocab_size={vocab_size}"
        )

    log_probs = F.log_softmax(shifted_logits.float(), dim=-1)

    token_logprobs = torch.gather(
        log_probs,
        dim=-1,
        index=shifted_labels.unsqueeze(-1),
    ).squeeze(-1)

    token_logprobs = token_logprobs.masked_fill(
        ~shifted_response_mask,
        0.0,
    )

    return token_logprobs, shifted_response_mask


def compute_group_advantages(
    rollouts: list[dict[str, Any]],
    reward_field: str = "reward",
) -> torch.Tensor:
    rewards = torch.tensor(
        [float(r[reward_field]) for r in rollouts],
        dtype=torch.float32,
    )

    group_ids = [int(r["group_id"]) for r in rollouts]

    advantages = torch.zeros_like(rewards)

    for group_id in sorted(set(group_ids)):
        indices = [i for i, gid in enumerate(group_ids) if gid == group_id]
        group_rewards = rewards[indices]

        mean = group_rewards.mean()
        std = group_rewards.std(unbiased=False)

        if std.item() < 1e-8:
            advantages[indices] = 0.0
        else:
            advantages[indices] = (group_rewards - mean) / (std + 1e-8)

    return advantages


# ============================================================
# SFT on golden reasoning traces
# ============================================================

def build_sft_target(row: pd.Series) -> str:
    reasoning_trace = str(row["reasoning_trace"]).strip()
    target_answer = str(row["target_answer"]).strip()

    if reasoning_trace.lower() == "nan":
        reasoning_trace = ""

    if "####" in reasoning_trace:
        return reasoning_trace

    if reasoning_trace:
        return f"{reasoning_trace}\n#### {target_answer}"

    return f"#### {target_answer}"


def build_sft_batch(
    tokenizer,
    rows: list[pd.Series],
    cfg: TrainConfig,
    device,
) -> dict[str, torch.Tensor]:
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for row in rows:
        question = str(row["question"])
        prompt = build_prompt(question, cfg.prompt_style)
        target = build_sft_target(row)

        prompt_ids = tokenizer(
            prompt + "\n",
            add_special_tokens=False,
            truncation=True,
            max_length=cfg.max_prompt_length,
        )["input_ids"]

        remaining_len = max(cfg.max_sft_length - len(prompt_ids), 1)

        target_ids = tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=remaining_len,
        )["input_ids"]

        if len(target_ids) == 0:
            target_ids = [tokenizer.eos_token_id]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id

    padded_input_ids = []
    padded_labels = []
    padded_attention_mask = []

    for input_ids, labels, attention_mask in zip(
        input_ids_list,
        labels_list,
        attention_mask_list,
    ):
        pad_len = max_len - len(input_ids)

        padded_input_ids.append(input_ids + [pad_id] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)
        padded_attention_mask.append(attention_mask + [0] * pad_len)

    labels_tensor = torch.tensor(
        padded_labels,
        dtype=torch.long,
        device=device,
    )

    if (labels_tensor != -100).sum().item() == 0:
        raise RuntimeError("SFT batch has no supervised target tokens.")

    return {
        "input_ids": torch.tensor(
            padded_input_ids,
            dtype=torch.long,
            device=device,
        ),
        "attention_mask": torch.tensor(
            padded_attention_mask,
            dtype=torch.long,
            device=device,
        ),
        "labels": labels_tensor,
    }


def compute_sft_loss(
    model,
    sft_batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model(
        input_ids=sft_batch["input_ids"],
        attention_mask=sft_batch["attention_mask"],
        labels=sft_batch["labels"],
    )

    loss = outputs.loss

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite SFT loss: {loss.detach().cpu().item()}")

    return loss


# ============================================================
# Training step
# ============================================================

def train_step_grpo_sft(
    model,
    tokenizer,
    optimizer,
    rollouts: list[dict[str, Any]],
    sft_rows: list[pd.Series],
    cfg: TrainConfig,
) -> dict[str, float]:
    model.train()

    advantages = compute_group_advantages(
        rollouts=rollouts,
        reward_field="reward",
    ).to(model.device)

    rollout_batch = build_logprob_batch(
        rollouts=rollouts,
        tokenizer=tokenizer,
        device=model.device,
    )

    with torch.no_grad():
        old_token_logprobs, response_mask = compute_token_logprobs(
            model=model,
            batch=rollout_batch,
        )
        old_token_logprobs = old_token_logprobs.detach()
        response_mask = response_mask.detach()

    new_token_logprobs, response_mask = compute_token_logprobs(
        model=model,
        batch=rollout_batch,
    )

    token_advantages = advantages.unsqueeze(1)

    log_ratio = new_token_logprobs - old_token_logprobs
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)

    ratio = torch.exp(log_ratio)

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - cfg.clip_eps,
        1.0 + cfg.clip_eps,
    )

    surrogate_1 = ratio * token_advantages
    surrogate_2 = clipped_ratio * token_advantages

    per_token_objective = torch.minimum(surrogate_1, surrogate_2)
    per_token_policy_loss = -per_token_objective

    masked_policy_loss = per_token_policy_loss.masked_fill(
        ~response_mask,
        0.0,
    )

    num_response_tokens = response_mask.sum().clamp_min(1).float()

    grpo_loss = masked_policy_loss.sum() / num_response_tokens

    sft_batch = build_sft_batch(
        tokenizer=tokenizer,
        rows=sft_rows,
        cfg=cfg,
        device=model.device,
    )

    sft_loss = compute_sft_loss(
        model=model,
        sft_batch=sft_batch,
    )

    loss = grpo_loss + cfg.sft_coef * sft_loss

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss: "
            f"total={loss.detach().cpu().item()}, "
            f"grpo={grpo_loss.detach().cpu().item()}, "
            f"sft={sft_loss.detach().cpu().item()}"
        )

    optimizer.zero_grad()
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        cfg.max_grad_norm,
    )

    if not torch.isfinite(grad_norm):
        raise RuntimeError(f"Non-finite grad norm: {grad_norm}")

    optimizer.step()

    mean_reward = sum(float(r["reward"]) for r in rollouts) / len(rollouts)
    mean_acc = sum(float(r["answer_correct"]) for r in rollouts) / len(rollouts)
    mean_format = sum(float(r["format_reward"]) for r in rollouts) / len(rollouts)

    response_lengths = response_mask.sum(dim=-1)

    mean_response_len = float(
        response_lengths.float().mean().detach().cpu().item()
    )

    mean_abs_advantage = float(
        advantages.float().abs().mean().detach().cpu().item()
    )

    valid_ratio = ratio[response_mask]
    if valid_ratio.numel() > 0:
        mean_ratio = float(valid_ratio.mean().detach().cpu().item())
        clip_fraction = float(
            ((valid_ratio < 1.0 - cfg.clip_eps) | (valid_ratio > 1.0 + cfg.clip_eps))
            .float()
            .mean()
            .detach()
            .cpu()
            .item()
        )
    else:
        mean_ratio = 1.0
        clip_fraction = 0.0

    return {
        "loss": float(loss.detach().cpu().item()),
        "grpo_loss": float(grpo_loss.detach().cpu().item()),
        "sft_loss": float(sft_loss.detach().cpu().item()),
        "mean_reward": mean_reward,
        "mean_answer_correct": mean_acc,
        "mean_format_reward": mean_format,
        "mean_response_len": mean_response_len,
        "mean_abs_advantage": mean_abs_advantage,
        "mean_ratio": mean_ratio,
        "clip_fraction": clip_fraction,
        "grad_norm": float(grad_norm.detach().cpu().item()),
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_greedy(
    model,
    tokenizer,
    df: pd.DataFrame,
    cfg: TrainConfig,
    max_eval: int = 50,
) -> dict[str, float]:
    model.eval()

    eval_df = df.iloc[: min(len(df), max_eval)]

    correct = 0

    for _, row in eval_df.iterrows():
        question = str(row["question"])
        gold_answer = str(row["target_answer"])
        prompt = build_prompt(question, cfg.prompt_style)

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_prompt_length,
        ).to(model.device)

        generation = model.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            remove_invalid_values=True,
        )

        response_ids = generation[0, enc["input_ids"].shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)

        pred = extract_final_answer(response)

        if cfg.debug:
            print("=" * 80, flush=True)
            print("EVAL QUESTION:", flush=True)
            print(question, flush=True)
            print("EVAL RESPONSE:", flush=True)
            print(response, flush=True)
            print("GOLD:", gold_answer, flush=True)
            print("PRED:", pred, flush=True)
            print("MATCH:", answers_match(pred, gold_answer), flush=True)
            print("=" * 80, flush=True)

        if answers_match(pred, gold_answer):
            correct += 1

    total = len(eval_df)
    acc = correct / total if total > 0 else 0.0

    model.train()

    return {
        "eval_accuracy": float(acc),
        "eval_correct": float(correct),
        "eval_total": float(total),
    }


# ============================================================
# Train loop
# ============================================================

def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = load_gsm8k_dataframe(
        csv_path=cfg.train_csv,
        max_rows=cfg.max_rows,
    )

    model, tokenizer = load_policy_model(
        model_name=cfg.model_name,
        device=cfg.device,
    )

    debug_model_after_loading(model, tokenizer, cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
    )

    eval_metrics = evaluate_greedy(
        model=model,
        tokenizer=tokenizer,
        df=df,
        cfg=cfg,
    )

    print(f"Before training: {eval_metrics}", flush=True)

    global_step = 0

    for epoch in range(cfg.epochs):
        shuffled_df = df.sample(
            frac=1,
            random_state=cfg.seed + epoch,
        ).reset_index(drop=True)

        for start in range(0, len(shuffled_df), cfg.batch_size):
            batch_df = shuffled_df.iloc[start : start + cfg.batch_size]

            questions = batch_df["question"].astype(str).tolist()
            gold_answers = batch_df["target_answer"].astype(str).tolist()

            prompts = [
                build_prompt(question, cfg.prompt_style)
                for question in questions
            ]

            rollouts = sample_rollouts_for_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                gold_answers=gold_answers,
                cfg=cfg,
            )

            sft_rows = [
                row
                for _, row in batch_df.iterrows()
            ]

            metrics = train_step_grpo_sft(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                rollouts=rollouts,
                sft_rows=sft_rows,
                cfg=cfg,
            )

            global_step += 1

            print(
                f"epoch={epoch + 1} "
                f"step={global_step} "
                f"loss={metrics['loss']:.4f} "
                f"grpo={metrics['grpo_loss']:.4f} "
                f"sft={metrics['sft_loss']:.4f} "
                f"reward={metrics['mean_reward']:.4f} "
                f"acc={metrics['mean_answer_correct']:.4f} "
                f"fmt={metrics['mean_format_reward']:.4f} "
                f"len={metrics['mean_response_len']:.1f} "
                f"abs_adv={metrics['mean_abs_advantage']:.4f} "
                f"ratio={metrics['mean_ratio']:.4f} "
                f"clip={metrics['clip_fraction']:.4f} "
                f"grad={metrics['grad_norm']:.4f}",
                flush=True,
            )

    eval_metrics = evaluate_greedy(
        model=model,
        tokenizer=tokenizer,
        df=df,
        cfg=cfg,
    )

    print(f"After training: {eval_metrics}", flush=True)

    model.save_pretrained(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)

    print(f"Saved model to: {cfg.out_dir}", flush=True)


# ============================================================
# CLI
# ============================================================

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Self-contained GRPO + auxiliary SFT training on GSM8K."
    )

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument(
        "--prompt-style",
        type=str,
        default=DEFAULT_PROMPT_STYLE,
        choices=sorted(PROMPT_TEMPLATES.keys()),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=200)

    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)

    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)

    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help="Use greedy generation for rollouts. Useful for debugging.",
    )

    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--format-reward", type=float, default=0.1)
    parser.add_argument("--clip-eps", type=float, default=0.2)

    parser.add_argument("--sft-coef", type=float, default=0.05)
    parser.add_argument("--max-sft-length", type=int, default=768)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-print-samples", type=int, default=0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return TrainConfig(
        model_name=args.model_name,
        train_csv=args.train_csv,
        out_dir=args.out_dir,
        prompt_style=args.prompt_style,
        seed=args.seed,
        max_rows=args.max_rows,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sampling,
        lr=args.lr,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        device=device,
        format_reward=args.format_reward,
        clip_eps=args.clip_eps,
        sft_coef=args.sft_coef,
        max_sft_length=args.max_sft_length,
        debug=args.debug,
        debug_print_samples=args.debug_print_samples,
    )


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()