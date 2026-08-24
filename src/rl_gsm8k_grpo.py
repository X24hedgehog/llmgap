from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import set_seed

from gsm8k_prompts import PROMPT_TEMPLATES, DEFAULT_PROMPT_STYLE

from rl_gsm8k_basic import (
    load_gsm8k_dataframe,
    build_prompt,
    load_policy_model,
    sample_rollouts_for_batch,
    build_logprob_batch,
    compute_group_advantages,
    evaluate_greedy,
)


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
    max_new_tokens: int = 512

    temperature: float = 0.8
    top_p: float = 0.95

    lr: float = 1e-7
    epochs: int = 1
    max_grad_norm: float = 0.5

    device: str = "cuda"

    format_reward: float = 0.1

    # GRPO/PPO-style clipping.
    clip_eps: float = 0.2

    # Not used in the first GRPO version.
    # Later we will use this for reference-model KL.
    beta_kl: float = 0.0




def compute_token_logprobs(
    model,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    logits = outputs.logits
    labels = batch["labels"]

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    log_probs = F.log_softmax(shifted_logits.float(), dim=-1)

    gather_labels = shifted_labels.clone()
    gather_labels[gather_labels == -100] = 0

    token_log_probs = log_probs.gather(
        dim=-1,
        index=gather_labels.unsqueeze(-1),
    ).squeeze(-1)

    response_mask = shifted_labels != -100

    token_log_probs = token_log_probs.masked_fill(
        ~response_mask,
        0.0,
    )

    return token_log_probs, response_mask


def train_step_grpo(
        model,
        tokenizer,
        optimizer,
        rollouts: list[dict[str, Any]],
        cfg: TrainConfig
) -> dict[str, float]:
    """
    One clipped GRPO update.

    Core idea:
        1. Compute group-relative advantages.
        2. Compute old token log-probs without gradient.
        3. Compute new token log-probs with gradient.
        4. Compute ratio = exp(new - old).
        5. Apply PPO/GRPO clipping.
        6. Optimize clipped policy-gradient loss.
    """

    model.train()

    advantages = compute_group_advantages(
        rollouts=rollouts,
        reward_field="reward",
    ).to(model.device)

    # Shape: [B]

    batch = build_logprob_batch(
        rollouts=rollouts,
        tokenizer=tokenizer,
        device=model.device
    )

    with torch.no_grad():
        old_token_logprobs, response_mask = compute_token_logprobs(
            model=model,
            batch=batch,
        ) # Shape: [B, T-1]

        old_token_logprobs = old_token_logprobs.detach()
        response_mask = response_mask.detach()
    
    new_token_logprobs , response_mask = compute_token_logprobs(
        model=model,
        batch=batch,
    ) # Shape: [B, T-1]

    token_advantages = advantages.unsqueeze(1) # Unsqueeze advantages shape [B] to [B, 1] for broadcasting with token log-probs.

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

    # Shape: [B, T-1]

    per_token_objective = torch.minimum(
        surrogate_1,
        surrogate_2,
    )

    per_token_loss = - per_token_objective

    masked_loss = per_token_loss.masked_fill(~response_mask, 0.0)
    num_response_tokens = response_mask.sum().clamp_min(1).float()
    loss = masked_loss.sum() / num_response_tokens

    if not torch.isfinite(loss):
        raise RuntimeError(f"Loss is not finite: {loss.detach().cpu().item()}")
    

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        cfg.max_grad_norm,
    )

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

    mean_ratio = float(
        ratio[response_mask].mean().detach().cpu().item()
    )

    clip_fraction = float(
        ((ratio < 1.0 - cfg.clip_eps) | (ratio > 1.0 + cfg.clip_eps))[response_mask]
        .float()
        .mean()
        .detach()
        .cpu()
        .item()
    )

    return {
        "loss": float(loss.detach().cpu().item()),
        "mean_reward": mean_reward,
        "mean_answer_correct": mean_acc,
        "mean_format_reward": mean_format,
        "mean_response_len": mean_response_len,
        "mean_abs_advantage": mean_abs_advantage,
        "mean_ratio": mean_ratio,
        "clip_fraction": clip_fraction,
    }


def train(cfg: TrainConfig) -> None:

    set_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)

    df = load_gsm8k_dataframe(
        csv_path = cfg.train_csv, 
        max_rows = cfg.max_rows
    )

    model, tokenizer = load_policy_model(
        model_name = cfg.model_name,    
        device = cfg.device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.lr
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

        shuffled_df = df.sample(frac=1, random_state=cfg.seed + epoch).reset_index(drop=True)

        for start in range (0, len(shuffled_df), cfg.batch_size):

            batch_df = shuffled_df.iloc[start: start + cfg.batch_size]

            questions = batch_df['question'].astype(str).tolist()
            gold_answers = batch_df['target_answer'].astype(str).tolist()

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

            metrics = train_step_grpo(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        rollouts=rollouts,
                        cfg=cfg,
                    )

            global_step += 1

            print(
                f"epoch={epoch + 1} "
                f"step={global_step} "
                f"loss={metrics['loss']:.4f} "
                f"reward={metrics['mean_reward']:.4f} "
                f"acc={metrics['mean_answer_correct']:.4f} "
                f"fmt={metrics['mean_format_reward']:.4f} "
                f"len={metrics['mean_response_len']:.1f} "
                f"abs_adv={metrics['mean_abs_advantage']:.4f} "
                f"ratio={metrics['mean_ratio']:.4f} "
                f"clip={metrics['clip_fraction']:.4f}",
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


def parse_args() -> TrainConfig:
    """
    Parse command-line arguments and create TrainConfig.
    """
    parser = argparse.ArgumentParser(
        description="Basic clipped GRPO training on GSM8K."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model name for the trainable policy.",
    )

    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to GSM8K-style CSV with columns question and target_answer.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory where the trained model will be saved.",
    )

    parser.add_argument(
        "--prompt-style",
        type=str,
        default=DEFAULT_PROMPT_STYLE,
        choices=sorted(PROMPT_TEMPLATES.keys()),
        help="Prompt template style from gsm8k_prompts.py.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=200,
        help="Maximum number of training rows to use for debugging.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of sampled responses per question.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of questions per RL update.",
    )

    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help="Maximum number of prompt tokens.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens per response.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for rollout generation.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling parameter.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-7,
        help="Learning rate.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of passes over the loaded training data.",
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Gradient clipping norm.",
    )

    parser.add_argument(
    "--clip-eps",
    type=float,
    default=0.2,
    help="GRPO/PPO clipping epsilon.",
    )

    parser.add_argument(
        "--beta-kl",
        type=float,
        default=0.0,
        help="KL coefficient against a frozen reference model. Not used in first GRPO version.",
    )

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
        lr=args.lr,
        epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        device=device,
        clip_eps=args.clip_eps,
        beta_kl=args.beta_kl,
    )

def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()