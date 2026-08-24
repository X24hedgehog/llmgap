from __future__ import annotations

import argparse
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from gsm8k_prompts import PROMPT_TEMPLATES, DEFAULT_PROMPT_STYLE


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

    lr: float = 1e-6
    epochs: int = 1
    max_grad_norm: float = 0.5

    device: str = "cuda"

    format_reward: float = 0.1



def load_gsm8k_dataframe(csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    if 'split' in df.columns:
        df = df[df['split'] == 'train'].copy()

    if max_rows is not None:
        df = df.head(max_rows).copy()

    return df.reset_index(drop=True)

# gsm8k_data_path = '/cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs_gsm8k.csv'
# df = load_gsm8k_dataframe(gsm8k_data_path, max_rows=200)
# print(df.head(5))
# print(df.columns)


def build_prompt(q: str, prompt_style: str = DEFAULT_PROMPT_STYLE) -> str:
    template = PROMPT_TEMPLATES[prompt_style]

    return template.format(question=q.strip())


def extract_last_number(text: str) -> str | None:
    """
    Extract the last number from the text.
    The number can be an integer or a decimal number.
    """
    text = str(text)
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if matches:
        return matches[-1]
    return None


def extract_final_answer(text: str) -> str:
    text = str(text)

    gsm8k_match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if gsm8k_match:
        return gsm8k_match.group(1).replace(",", "").strip()

    final_phrase_match = re.search(
        r"(?:final answer|answer is|answer:|therefore|so)\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if final_phrase_match:
        return final_phrase_match.group(1).replace(",", "").strip()

    final_answer = extract_last_number(text)
    if final_answer is not None:
        return final_answer.replace(",", "").strip()

    return text.strip()
    

def normalize_answer(text: str) -> str:
    """
    Normalize answer string before comparison.
    """
    text = str(text).strip().lower()
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .")
    return text


def answers_match(pred: str, gold: str) -> bool:
    """
    Compare predicted answer and gold answer.

    This is robust to gold being either:
        - clean answer, e.g. "42"
        - GSM8K-style answer, e.g. "#### 42"
        - full solution ending with "#### 42"
    """
    pred_norm = normalize_answer(pred)

    gold_extracted = extract_final_answer(gold)
    gold_norm = normalize_answer(gold_extracted)

    try:
        pred_float = float(pred_norm)
        gold_float = float(gold_norm)
        return abs(pred_float - gold_float) < 1e-9
    except ValueError:
        return pred_norm == gold_norm


def compute_correct_answer_reward(response: str, gold_answer: str) -> float:
    pred_answer = extract_final_answer(response)
    return 1.0 if answers_match(pred_answer, gold_answer) else 0.0


# responses = [
#     "We compute 84 / 2 = 42.\n#### 42",
#     "We compute 84 / 2 = 43.\n#### 43",
#     "Julie should read 42.0 pages.",
#     "I do not know.",
# ]

# gold = "42"

# for r in responses:
#     print("response:", r)
#     print("extracted:", extract_final_answer(r))
#     print("reward:", compute_correct_answer_reward(r, gold))
#     print()

def compute_format_reward(response: str, format_reward: float = 0.1) -> float:
    text = str(response)

    match = re.search(r"####\s*[-+]?\d[\d,]*(?:\.\d+)?",text)

    return format_reward if match else 0.0


def compute_total_reward(response: str, gold_answer: str) -> dict[str, float]:
    correct_reward = compute_correct_answer_reward(response, gold_answer)
    format_reward = compute_format_reward(response)

    total_reward = correct_reward + format_reward

    return {
        "reward": total_reward,
        "answer_correct": correct_reward,
        "format_reward": format_reward,
    }


def load_policy_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    model.to(device)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.train()

    return model, tokenizer


def sample_one_response(
    model,
    tokenizer,
    prompt: str,
    cfg: TrainConfig,
) -> dict[str, Any]:
    enc = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_prompt_length)
    
    enc = {k: v.to(model.device) for k, v in enc.items()}
    # Ex: enc = {"input_ids": tensor([[151644, 872, 198, ...]]), "attention_mask": tensor([[1, 1, 1, ...]])}

    prompt_len = enc["input_ids"].shape[1]
    # enc["input_ids"] shape: [batch_size, sequence_length]

    model.eval()
    with torch.no_grad():
        generation = model.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=max(float(cfg.temperature), 1e-5),
            top_p=float(cfg.top_p),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            remove_invalid_values=True,
            renormalize_logits=True,
        )
    # generation shape: [batch_size, prompt_length + generated_length] -> generation[0]: first (and only) generated sequence of token ids (batch_size = 1 here)

    full_ids = generation[0].detach().cpu().tolist() # Moving to CPU allows us to convert the tensor to a list of token IDs and then decode it into a string

    response_ids = full_ids[prompt_len:]  # Generated token IDs after the prompt

    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip() 

    return {
        'prompt': prompt,
        'response': response_text,
        'full_ids': full_ids,
        'response_start': prompt_len
    }

def sample_rollouts_for_batch(
    model,
    tokenizer,
    prompts: list[str],
    gold_answers: list[str],
    cfg: TrainConfig,
) -> list[dict[str, Any]]:
    """
    For each prompt in the batch, sample cfg.num_samples responses.

    Each prompt is one group.
    Later, group_id is used to compute group-relative advantages.
    """

    rollouts = []
    
    for group_id, (prompt, gold) in enumerate(zip(prompts, gold_answers)):

        for sample_idx in range(cfg.num_samples):

            rollout = sample_one_response(model, tokenizer, prompt, cfg)
            rollout['group_id'] = group_id
            rollout['sample_idx'] = sample_idx
            rollout['gold_answer'] = gold

            reward_info = compute_total_reward(
                response=rollout['response'], 
                gold_answer=gold)

            rollout.update(reward_info) # Fields added: "reward" (= answer_correct + format_reward), "answer_correct" (binary), "format_reward" (weight 0.1)
            rollouts.append(rollout)

            if group_id == 0 and sample_idx < 2:
                print("=" * 80)
                print("PROMPT:")
                print(prompt)
                print("RESPONSE:")
                print(rollout["response"])
                print("GOLD:", gold)
                print("EXTRACTED:", extract_final_answer(rollout["response"]))
                print("REWARD_INFO:", reward_info)
                print("=" * 80)

    assert len(rollouts) == len(prompts) * cfg.num_samples

    return rollouts

    # Example of 1 rollout dict:
    # {
    #     'prompt': 'Question: ...',    
    #     'response': 'Answer: ...',  
    #     'full_ids': [151644, 872, 198, ...],  # token IDs of prompt + response
    #     'response_start': 10,  # index where the response starts in full
    #     'group_id': 0,
    #     'sample_idx': 0,
    #     'gold_answer': '42',
    #     'reward': 1.0,
    #     'answer_correct': 1.0,
    #     'format_reward': 0.1
    # }



def build_logprob_batch(
    rollouts: list[dict[str, Any]],
    tokenizer,
    device,
) -> dict[str, torch.Tensor]:
    """
    Build a batch of log-probs for the sampled rollouts.

    Each rollout contains:
        - full_ids: the token IDs of the prompt + response
        - response_start: the index where the response starts in full_ids
    """

    # Example of 1 rollout dict:
    # {
    #     'prompt': 'Question: ...',    
    #     'response': 'Answer: ...',
    #     'full_ids': [151644, 872, 198, ...],  # token IDs of prompt + response
    #     'response_start': 10,  # index where the response starts in full
    #     'group_id': 0,
    #     'sample_idx': 0,
    #     'gold_answer': '42',
    #     'reward': 1.0,
    #     'answer_correct': 1.0,
    #     'format_reward': 0.1
    # }

    input_ids = []
    attention_mask = []
    labels = []

    # Extract full_ids from each rollout
    full_ids_list = [list(r["full_ids"]) for r in rollouts]

    # Extract response_start indices from each rollout
    response_starts = [int(r['response_start']) for r in rollouts]

    max_len = max([len(ids) for ids in full_ids_list])

    pad_id = tokenizer.pad_token_id

    for full_ids, response_start in zip(full_ids_list, response_starts):

        pad_len = max_len - len(full_ids)

        padded_ids = full_ids + [pad_id] * pad_len  # Pad the sequence to max_len
        mask = [1] * len(full_ids) + [0] * pad_len  # Attention mask: 1 for real tokens, 0 for padding
        label = [-100] * response_start + full_ids[response_start:] + [-100] * pad_len  # Labels: -100 for prompt tokens and padding, token IDs

        input_ids.append(padded_ids)
        attention_mask.append(mask)
        labels.append(label)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long, device=device),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long, device=device),
        'labels': torch.tensor(labels, dtype=torch.long, device=device)
    }

    # Example output:
    # {
    #     'input_ids': tensor([[151644, 872, 198, ..., 0, 0, 0],
    #                          [151644, 872, 198, ..., 0, 0, 0]]),
    #     'attention_mask': tensor([[1, 1, 1, ..., 0, 0, 0],
    #                               [1, 1, 1, ..., 0, 0, 0]]),
    #     'labels': tensor([[-100, -100, -100, ..., -100, -100, -100],
    #                       [-100, -100, -100, ..., -100, -100, -100]])
    # }

    # Shape: 
    # input_ids: [batch_size, max_seq_len], 
    # attention_mask: [batch_size, max_seq_len], 
    # labels: [batch_size, max_seq_len]


def compute_sequence_logprobs(
    model,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log pi_theta(response | prompt) for each rollout.

    Returns:
        seq_logprobs:
            Tensor of shape [batch_size_rollouts]

        response_lengths:
            Tensor of shape [batch_size_rollouts]


    The batch contains:
        - input_ids: token IDs of prompt + response, padded to max length
        - attention_mask: mask for input_ids
        - labels: token IDs of response with -100 for prompt and padding
)
    """

    # Example of batch:
    # {
    #     'input_ids': tensor([[151644, 872, 198, ..., 0, 0, 0],
    #                          [151644, 872, 198, ..., 0, 0, 0]]),
    #     'attention_mask': tensor([[1, 1, 1, ..., 0, 0, 0],
    #                               [1, 1, 1, ..., 0, 0, 0]]),
    #     'labels': tensor([[-100, -100, -100, ..., -100, -100, -100],
    #                       [-100, -100, -100, ..., -100, -100, -100]])
    # }

    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
    )

    # outputs's fields include:
    # - loss: the cross-entropy loss averaged over all non-ignored tokens in the batch
    # - logits: the raw output scores (before softmax) for each token in the vocabulary, with shape [batch_size, seq_len, vocab_size]
    # - hidden_states: (optional)

    logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

    labels = batch['labels']  # Shape: [batch_size, seq_len]

    shifted_logits = logits[..., :-1, :].contiguous()  # Shift logits to align with labels 
    # Shape: [batch_size, seq_len - 1, vocab_size]

    shifted_labels = labels[..., 1:].contiguous()  # Shift labels to align with logits
    # Shape: [batch_size, seq_len - 1]

    log_probs = F.log_softmax(shifted_logits.float(), dim=-1)  # Shape: [batch_size, seq_len - 1, vocab_size]

    gather_labels = shifted_labels.clone()
    gather_labels[gather_labels == -100] = 0  # Replace -100 with 0 for gathering log_probs

    token_log_probs = log_probs.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_size, seq_len - 1]

    response_mask = (shifted_labels != -100)  # Mask to identify response tokens (True for response tokens, False for prompt and padding)
    # Shape: [batch_size, seq_len - 1]

    token_log_probs = token_log_probs.masked_fill(
        ~response_mask,
        0.0
    )  
    # Zero out log-probs for prompt and padding tokens
    # Shape: [batch_size, seq_len - 1]

    seq_logprobs = token_log_probs.sum(dim=-1)  # Sum log-probs over response tokens for each rollout
    # Shape: [batch_size]

    response_lengths = response_mask.sum(dim=-1).clamp_min(1)  # Count the number of response tokens for each rollout
    # Shape: [batch_size]

    return seq_logprobs, response_lengths


def compute_group_advantages(
    rollouts: list[dict[str, Any]],
    reward_field: str = "reward",
) -> torch.Tensor:
    advantages = [0.0 for _ in rollouts]

    grouped: dict[int, list[int]] = {}

    for idx, rollout in enumerate(rollouts):
        group_id = int(rollout["group_id"])
        grouped.setdefault(group_id, []).append(idx)

    for group_id, indices in grouped.items():
        rewards = torch.tensor(
            [float(rollouts[i][reward_field]) for i in indices],
            dtype=torch.float32,
        )

        mean = rewards.mean()
        std = rewards.std(unbiased=False)

        if std.item() < 1e-6:
            group_advantages = torch.zeros_like(rewards)
        else:
            group_advantages = (rewards - mean) / (std + 1e-6)

        for local_idx, rollout_idx in enumerate(indices):
            advantages[rollout_idx] = float(group_advantages[local_idx])

    return torch.tensor(advantages, dtype=torch.float32)


def train_step_reinforce(
    model,
    tokenizer,
    optimizer,
    rollouts: list[dict[str, Any]],
    cfg: TrainConfig,
) -> dict[str, float]:
    """
    One group-normalized REINFORCE update.

    Loss:
        loss = - advantage * log_prob(response | prompt)

    We normalize sequence log-prob by response length.
    """
    model.train()

    advantages = compute_group_advantages(
        rollouts=rollouts,
        reward_field="reward",
    ).to(model.device)

    batch = build_logprob_batch(
        rollouts=rollouts,
        tokenizer=tokenizer,
        device=model.device,
    )

    seq_logprobs, response_lengths = compute_sequence_logprobs(
        model=model,
        batch=batch,
    )

    normalized_logprobs = seq_logprobs / response_lengths

    loss = -(advantages * normalized_logprobs).mean()

    print("advantages:", advantages.detach().cpu())
    print("seq_logprobs:", seq_logprobs.detach().cpu())
    print("response_lengths:", response_lengths.detach().cpu())
    print("normalized_logprobs:", normalized_logprobs.detach().cpu())
    print("loss:", loss.detach().cpu())

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss.detach().cpu().item()}")

    optimizer.zero_grad()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise RuntimeError(f"Non-finite gradient in parameter: {name}")

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        cfg.max_grad_norm,
    )

    optimizer.step()

    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(f"Non-finite parameter after optimizer step: {name}")

    mean_reward = sum(float(r["reward"]) for r in rollouts) / len(rollouts)
    mean_acc = sum(float(r["answer_correct"]) for r in rollouts) / len(rollouts)
    mean_format = sum(float(r["format_reward"]) for r in rollouts) / len(rollouts)

    mean_response_len = float(
        response_lengths.float().mean().detach().cpu().item()
    )

    mean_advantage = float(
        advantages.float().mean().detach().cpu().item()
    )

    mean_abs_advantage = float(
        advantages.float().abs().mean().detach().cpu().item()
    )

    return {
        "loss": float(loss.detach().cpu().item()),
        "mean_reward": mean_reward,
        "mean_answer_correct": mean_acc,
        "mean_format_reward": mean_format,
        "mean_response_len": mean_response_len,
        "mean_advantage": mean_advantage,
        "mean_abs_advantage": mean_abs_advantage,
    }


@torch.no_grad()
def evaluate_greedy(
    model,
    tokenizer,
    df: pd.DataFrame,
    cfg: TrainConfig,
    max_eval_rows: int=50,
) -> dict[str, float]:
    """
    Evaluate the model with greedy decoding.

    This is a quick way to check if the model is improving during training.
    """

    model.eval()

    eval_df = df.head(max_eval_rows)

    correct, total = 0, 0

    for _, row in eval_df.iterrows():
        question = str(row['question'])
        gold = row['target_answer']

        prompt = build_prompt(question, cfg.prompt_style)

        enc = tokenizer(prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=cfg.max_prompt_length
                        ).to(model.device)
        
        # Example of enc: 
        # {
        #     'input_ids': tensor([[  101,  ...]]),
        #     'attention_mask': tensor([[1, 1, 1, ...]])
        # }


        enc = {k: v.to(model.device) for k, v in enc.items()}

        prompt_len = enc["input_ids"].shape[1]

        generation = model.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        # generation shape: [1, prompt_len + generated_len]

        full_ids = generation[0].detach().cpu().tolist()

        response_ids = full_ids[prompt_len:]  # Generated token IDs after the prompt

        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        reward = compute_correct_answer_reward(response, gold)

        correct += reward

        total += 1

    accuracy = correct / max(total, 1)

    return {
        "eval_accuracy": accuracy,
        "eval_correct": float(correct),
        "eval_total": float(total),
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

            metrics = train_step_reinforce(
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
                f"abs_adv={metrics.get('mean_abs_advantage', 0.0):.4f}",
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
        description="Basic group-normalized REINFORCE training on GSM8K."
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
        default=1e-6,
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
    )

def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()