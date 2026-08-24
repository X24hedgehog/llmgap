#!/usr/bin/env python3
"""Stabilized ORM-based sampling and RL post-training.

This is a safer variant of orm_main.py with additional training stabilizers:
- Robust RLHFlow ORM decoding via mask-token position detection (no fixed -3 index).
- Anchored reward for train mode:
    reward = answer_correct + 0.1 * format_ok + 0.1 * orm_score
- Optional reward clipping + global reward normalization.
- Group filtering by reward range threshold.
- Optional KL penalty to a frozen reference model.
- Safer default hyperparameters for initial RL debugging.

Modes:
1) mode=rerank
2) mode=train
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from prm_main import (
    RERANK_ALIASES,
    SUPPORTED_TASKS,
    _build_val_data,
    _compute_group_advantages,
    _compute_sequence_logprobs,
    _evaluate_saved_checkpoints,
    _load_dataframe,
    _load_policy_model,
    _load_policy_tokenizer,
    _make_quant_config,
    _needs_4bit,
    _row_metadata,
    _sample_rollouts,
    _save_rollouts_csv,
    _select_best_rollouts,
    _set_seed,
    score_correct_answer,
)


DEFAULT_ORM_MODEL = "RLHFlow/Llama3.1-8B-ORM-Mistral-Data"
DEFAULT_ARMO_MODEL = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
ROOT_DIR = Path(__file__).resolve().parents[1]
USER_SCRATCH_ROOT = Path("/cluster/scratch") / Path.home().name
DEFAULT_OUTPUT_ROOT = USER_SCRATCH_ROOT / ROOT_DIR.name / "output" / "orm"


def _require_scratch_path(path_value: str | None, arg_name: str) -> None:
    if not path_value:
        return
    given = Path(path_value).expanduser()
    resolved = (Path.cwd() / given).resolve() if not given.is_absolute() else given.resolve()
    root = USER_SCRATCH_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"{arg_name} must be under {root}, got {resolved}. "
            "Use a /cluster/scratch/<user>/... output path."
        )


def _is_within_path(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _resolve_output_path(path_str: str, description: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = DEFAULT_OUTPUT_ROOT / path
    if not _is_within_path(path, USER_SCRATCH_ROOT):
        raise ValueError(f"{description} must be inside scratch ({USER_SCRATCH_ROOT}), got {path}")
    return path


def _finalize_output_args(args) -> None:
    if args.out_path:
        args.out_path = str(_resolve_output_path(args.out_path, "--out-path"))
    if args.selected_out_path:
        args.selected_out_path = str(_resolve_output_path(args.selected_out_path, "--selected-out-path"))
    if args.out_dir:
        args.out_dir = str(_resolve_output_path(args.out_dir, "--out-dir"))
    if args.rollout_out_dir:
        args.rollout_out_dir = str(_resolve_output_path(args.rollout_out_dir, "--rollout-out-dir"))


class BaseOutcomeRewardModel:
    def score_responses(
        self,
        prompts: list[str],
        responses: list[str],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class RLHFlowMathOutcomeRewardModel(BaseOutcomeRewardModel):
    """RLHFlow math ORM with mask-token label-position decoding.

    For each (prompt, response), build two conversations:
    - real: assistant label is "+"
    - mask: assistant label is dummy token "ки"

    We locate dummy-token positions from the mask conversation, then read
    P("+") from logits one position earlier (causal shift).
    """

    def __init__(self, model_name: str, load_in_4bit: bool) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        quant_config = _make_quant_config(load_in_4bit)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        plus_tag_id = self.tokenizer.encode("+", add_special_tokens=False)
        minus_tag_id = self.tokenizer.encode("-", add_special_tokens=False)
        if len(plus_tag_id) != 1 or len(minus_tag_id) != 1:
            raise ValueError(
                f"RLHFlow ORM expects single-token +/- labels, got {plus_tag_id} and {minus_tag_id}"
            )
        self.candidate_tokens = [plus_tag_id[0], minus_tag_id[0]]
        # Same dummy token used in PRM path, used only for position localization.
        self.special_tok_id = int(self.tokenizer("ки", return_tensors="pt").input_ids[0, 1])
        self.is_mistral_data = "Mistral-Data" in model_name

    def _format_user_turn(self, prompt: str, response: str) -> str:
        text = response.strip()
        if self.is_mistral_data:
            text = text.replace(" ки", "")
        return f"{prompt.strip()} {text}".strip()

    def _build_conversation_pair(
        self,
        prompt: str,
        response: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        user_content = self._format_user_turn(prompt, response)
        conversation = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "+"},
        ]
        conversation_mask = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "ки"},
        ]
        return conversation, conversation_mask

    def score_responses(
        self,
        prompts: list[str],
        responses: list[str],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        conversation_pairs = [
            self._build_conversation_pair(prompt, response)
            for prompt, response in zip(prompts, responses)
        ]
        results: list[dict[str, Any]] = []

        for start in range(0, len(conversation_pairs), batch_size):
            batch_pairs = conversation_pairs[start : start + batch_size]
            conversations = [pair[0] for pair in batch_pairs]
            conversations_mask = [pair[1] for pair in batch_pairs]

            input_ids = self.tokenizer.apply_chat_template(
                conversations,
                padding=True,
                return_tensors="pt",
                return_dict=False,
            ).to(self.model.device)
            input_ids_mask = self.tokenizer.apply_chat_template(
                conversations_mask,
                padding=True,
                return_tensors="pt",
                return_dict=False,
            ).to(self.model.device)

            if input_ids.shape != input_ids_mask.shape:
                raise ValueError(
                    f"ORM mask conversation shape {tuple(input_ids_mask.shape)} does not match "
                    f"label conversation shape {tuple(input_ids.shape)}"
                )

            with torch.no_grad():
                logits = self.model(input_ids).logits[:, :, self.candidate_tokens]
                probs_plus = logits.softmax(dim=-1)[:, :, 0]

            for row_idx in range(input_ids.shape[0]):
                label_mask = input_ids_mask[row_idx, 1:] == self.special_tok_id
                plus_probs = probs_plus[row_idx, :-1][label_mask].float().detach().cpu().tolist()
                if len(plus_probs) != 1:
                    raise ValueError(
                        f"Expected exactly one ORM label position, got {len(plus_probs)} "
                        f"for sample index {start + row_idx}"
                    )
                scalar = float(plus_probs[0])
                results.append({"reward": scalar, "orm_score": scalar})

        return results


class ArmoOutcomeRewardModel(BaseOutcomeRewardModel):
    """RLHFlow ArmoRM scoring via the published output.score API."""

    def __init__(self, model_name: str, load_in_4bit: bool) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        quant_config = _make_quant_config(load_in_4bit)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def score_responses(
        self,
        prompts: list[str],
        responses: list[str],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_responses = responses[start : start + batch_size]
            for prompt, response in zip(batch_prompts, batch_responses):
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    return_dict=False,
                ).to(self.model.device)
                with torch.no_grad():
                    output = self.model(input_ids)
                    score = float(output.score.float().item())
                results.append({"reward": score, "orm_score": score})
        return results


def build_outcome_reward_model(args) -> BaseOutcomeRewardModel:
    if args.orm_backend == "rlhflow_math_orm":
        return RLHFlowMathOutcomeRewardModel(
            args.orm_model_name,
            load_in_4bit=args.orm_load_in_4bit,
        )
    if args.orm_backend == "armo":
        return ArmoOutcomeRewardModel(
            args.orm_model_name,
            load_in_4bit=args.orm_load_in_4bit,
        )
    raise ValueError(f"Unsupported --orm-backend: {args.orm_backend}")


def _attach_orm_scores(
    orm: BaseOutcomeRewardModel,
    rollouts: list[dict[str, Any]],
    batch_size: int,
) -> None:
    results = orm.score_responses(
        [rollout["prompt"] for rollout in rollouts],
        [rollout["response"] for rollout in rollouts],
        batch_size=batch_size,
    )
    for rollout, orm_result in zip(rollouts, results):
        rollout.update(orm_result)


def _save_selected_csv(
    df: pd.DataFrame,
    prompts: list[str],
    golds: list[str],
    prompt_styles: list[str],
    best_rollouts: list[dict[str, Any]],
    scores: list[int],
    out_path: Path,
) -> None:
    rows = []
    for row_idx, (best_rollout, score) in enumerate(zip(best_rollouts, scores)):
        row = df.iloc[row_idx]
        rows.append(
            {
                "row_idx": row_idx,
                "question_id": row.get("question_id"),
                "split": row.get("split"),
                "prompt_style": prompt_styles[row_idx],
                "question": row.get("question"),
                "prompt": prompts[row_idx],
                "gold_answer": golds[row_idx],
                "selected_response": best_rollout["response"],
                "selected_reward": best_rollout.get("reward"),
                "selected_orm_score": best_rollout.get("orm_score"),
                "score": score,
                "correct": "Yes" if score else "No",
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def _extract_last_number(text: str) -> str | None:
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        return nums[-1]
    return None


def _normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"^(answer:|final answer:|the answer is|therefore)\s*", "", text)
    text = text.replace("$", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def _is_numeric_like(text: str) -> bool:
    return bool(re.fullmatch(r"[-+]?((\d+\.?\d*)|(\.\d+))", str(text).strip()))


def _extract_prediction_answer(response: str) -> str:
    boxed = _extract_boxed(response)
    if boxed is not None:
        return boxed

    final_answer_match = re.search(
        r"(?:final answer|answer)\s*(?:is|:)?\s*([^\n]+)",
        response,
        flags=re.IGNORECASE,
    )
    if final_answer_match:
        return final_answer_match.group(1).strip()

    last_num = _extract_last_number(response)
    if last_num is not None:
        return last_num

    return response.strip()


def _compute_answer_correct(response: str, gold: str, prompt_style: str) -> float:
    pred_ans = _extract_prediction_answer(response)
    pred_norm = _normalize_answer(pred_ans)
    gold_norm = _normalize_answer(gold)

    # For non-numeric styles (for example EEDI answer text), use normalized exact match.
    if not _is_numeric_like(gold_norm) or prompt_style == "eedi_correct_answer":
        return 1.0 if pred_norm == gold_norm else 0.0

    if not _is_numeric_like(pred_norm):
        return 0.0

    try:
        return 1.0 if abs(float(pred_norm) - float(gold_norm)) < 1e-9 else 0.0
    except ValueError:
        return 0.0


def _compute_format_reward(response: str) -> float:
    text = str(response)
    has_boxed = _extract_boxed(text) is not None
    has_final_answer_phrase = bool(re.search(r"\b(final answer|answer:)\b", text, flags=re.IGNORECASE))
    has_think_structure = "\n" in text and len(text.strip()) > 0
    return 1.0 if (has_boxed or has_final_answer_phrase or has_think_structure) else 0.0


def _apply_combined_rewards(
    rollouts: list[dict[str, Any]],
    gold_answers: list[str],
    prompt_styles: list[str],
    w_correct: float,
    w_format: float,
    w_orm: float,
    clip_min: float | None,
    clip_max: float | None,
    normalize: bool,
) -> None:
    rewards_for_norm = []
    for rollout in rollouts:
        group_id = int(rollout["group_id"])
        gold = gold_answers[group_id]
        prompt_style = prompt_styles[group_id]
        answer_correct = _compute_answer_correct(rollout["response"], gold, prompt_style)
        format_ok = _compute_format_reward(rollout["response"])
        orm_score = float(rollout.get("orm_score", 0.0))

        raw_reward = (w_correct * answer_correct) + (w_format * format_ok) + (w_orm * orm_score)
        if clip_min is not None:
            raw_reward = max(float(clip_min), raw_reward)
        if clip_max is not None:
            raw_reward = min(float(clip_max), raw_reward)

        rollout["answer_correct"] = answer_correct
        rollout["format_ok"] = format_ok
        rollout["raw_reward"] = float(raw_reward)
        rollout["reward"] = float(raw_reward)
        rewards_for_norm.append(float(raw_reward))

    if not normalize or not rewards_for_norm:
        for rollout in rollouts:
            rollout["reward_norm"] = float(rollout["reward"])
        return

    reward_tensor = torch.tensor(rewards_for_norm, dtype=torch.float32)
    mean = float(reward_tensor.mean().item())
    std = float(reward_tensor.std(unbiased=False).item())
    denom = std + 1e-6
    for rollout in rollouts:
        rollout["reward_norm"] = (float(rollout["reward"]) - mean) / denom


def _filter_groups_by_reward_range(
    rollouts: list[dict[str, Any]],
    reward_field: str,
    min_range: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for rollout in rollouts:
        grouped.setdefault(int(rollout["group_id"]), []).append(rollout)

    total_groups = len(grouped)
    kept_rollouts: list[dict[str, Any]] = []
    dropped_groups = 0
    per_group_ranges = []

    for _, group_rollouts in grouped.items():
        values = [float(rollout[reward_field]) for rollout in group_rollouts]
        reward_range = max(values) - min(values)
        per_group_ranges.append(reward_range)
        if reward_range > min_range:
            kept_rollouts.extend(group_rollouts)
        else:
            dropped_groups += 1

    kept_groups = total_groups - dropped_groups
    filtered_pct = (100.0 * dropped_groups / total_groups) if total_groups > 0 else 0.0
    mean_group_range = sum(per_group_ranges) / max(len(per_group_ranges), 1)

    stats = {
        "total_groups": float(total_groups),
        "kept_groups": float(kept_groups),
        "dropped_groups": float(dropped_groups),
        "filtered_pct": filtered_pct,
        "mean_group_reward_range": mean_group_range,
    }
    return kept_rollouts, stats


def _collect_rollout_metrics(
    rollouts: list[dict[str, Any]],
    reward_field: str,
    mean_kl: float,
    filtered_pct: float,
) -> dict[str, float]:
    if not rollouts:
        return {
            "mean_reward": 0.0,
            "mean_raw_reward": 0.0,
            "mean_answer_correct": 0.0,
            "mean_format_ok": 0.0,
            "mean_orm_score": 0.0,
            "mean_group_var": 0.0,
            "avg_response_len": 0.0,
            "mean_kl": mean_kl,
            "filtered_groups_pct": filtered_pct,
        }

    grouped: dict[int, list[dict[str, Any]]] = {}
    for rollout in rollouts:
        grouped.setdefault(int(rollout["group_id"]), []).append(rollout)

    group_vars = []
    for group_rollouts in grouped.values():
        vals = torch.tensor([float(r[reward_field]) for r in group_rollouts], dtype=torch.float32)
        group_vars.append(float(vals.var(unbiased=False).item()))

    metrics = {
        "mean_reward": sum(float(r[reward_field]) for r in rollouts) / len(rollouts),
        "mean_raw_reward": sum(float(r.get("raw_reward", 0.0)) for r in rollouts) / len(rollouts),
        "mean_answer_correct": sum(float(r.get("answer_correct", 0.0)) for r in rollouts) / len(rollouts),
        "mean_format_ok": sum(float(r.get("format_ok", 0.0)) for r in rollouts) / len(rollouts),
        "mean_orm_score": sum(float(r.get("orm_score", 0.0)) for r in rollouts) / len(rollouts),
        "mean_group_var": sum(group_vars) / max(len(group_vars), 1),
        "avg_response_len": sum(max(1, len(r.get("response_token_ids", []))) for r in rollouts) / len(rollouts),
        "mean_kl": mean_kl,
        "filtered_groups_pct": filtered_pct,
    }
    return metrics


def rerank_with_orm(args) -> None:
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(
            f"ORM scoring currently supports only {sorted(SUPPORTED_TASKS)}; got task={args.task}."
        )

    df = _load_dataframe(args.data_csv, args.task, split=args.split, max_rows=args.max_rows)
    prompts = [_build_prompt(row, args.task) for _, row in df.iterrows()]
    metadata = [_row_metadata(row) for _, row in df.iterrows()]

    tokenizer = _load_policy_tokenizer(args.model_name)
    load_4bit = args.load_in_4bit or _needs_4bit(args.model_name)
    quant_config = _make_quant_config(load_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if not load_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )
    rollouts = _sample_rollouts(
        model,
        tokenizer,
        prompts,
        metadata,
        task=args.task,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    del model
    torch.cuda.empty_cache()

    orm = build_outcome_reward_model(args)
    _attach_orm_scores(orm, rollouts, batch_size=args.orm_batch_size)

    out_path = Path(args.out_path)
    _save_rollouts_csv(rollouts, out_path)
    print(f"Saved {len(rollouts)} scored rollouts to {out_path}")

    best_rollouts = _select_best_rollouts(rollouts, reward_field=args.reward_field)
    prompts_eval, golds_eval, prompt_styles_eval = _build_val_data(df, task=args.task)
    predictions = [rollout["response"] for rollout in best_rollouts]
    scores = score_correct_answer(
        predictions,
        golds_eval,
        prompts_eval,
        prompt_styles_eval,
        batch_size=args.eval_batch_size,
    )
    accuracy = sum(scores) / len(scores) if scores else 0.0

    selected_out_path = Path(args.selected_out_path)
    _save_selected_csv(
        df,
        prompts_eval,
        golds_eval,
        prompt_styles_eval,
        best_rollouts,
        scores,
        selected_out_path,
    )
    print(f"Saved ORM-selected best-of-N responses to {selected_out_path}")
    print(f"Best-of-{args.num_samples} accuracy: {accuracy:.1%} ({sum(scores)}/{len(scores)})")


def train_with_orm(args) -> None:
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(
            f"ORM RL currently supports only {sorted(SUPPORTED_TASKS)}; got task={args.task}."
        )

    df = _load_dataframe(args.train_csv, args.task)
    if "split" in df.columns:
        df = df[df["split"] == "train"].reset_index(drop=True)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = _load_policy_tokenizer(args.model_name)
    load_4bit = args.load_in_4bit or _needs_4bit(args.model_name)
    model, quant_config = _load_policy_model(
        args.model_name,
        load_in_4bit=load_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    reference_model = None
    if args.kl_coef > 0:
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if not load_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        print(f"Loaded frozen reference model for KL regularization (kl_coef={args.kl_coef})")

    orm = build_outcome_reward_model(args)
    val_prompts, val_golds, val_prompt_styles = _build_val_data(val_df, task=args.task)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_df = train_df.sample(frac=1, random_state=args.seed + epoch).reset_index(drop=True)
        progress = range(0, len(epoch_df), args.batch_size)

        for start in progress:
            batch_df = epoch_df.iloc[start : start + args.batch_size].reset_index(drop=True)
            prompts = [_build_prompt(row, args.task) for _, row in batch_df.iterrows()]
            metadata = [_row_metadata(row) for _, row in batch_df.iterrows()]
            gold_answers = [str(row["target_answer"]) for _, row in batch_df.iterrows()]
            prompt_styles = [str(row.get("prompt_style", "")) for _, row in batch_df.iterrows()]

            rollouts = _sample_rollouts(
                model,
                tokenizer,
                prompts,
                metadata,
                task=args.task,
                num_samples=args.num_samples,
                batch_size=len(prompts),
                temperature=args.temperature,
                top_p=args.top_p,
            )
            _attach_orm_scores(orm, rollouts, batch_size=args.orm_batch_size)

            _apply_combined_rewards(
                rollouts,
                gold_answers=gold_answers,
                prompt_styles=prompt_styles,
                w_correct=args.w_answer_correct,
                w_format=args.w_format,
                w_orm=args.w_orm,
                clip_min=args.reward_clip_min,
                clip_max=args.reward_clip_max,
                normalize=args.normalize_rewards,
            )

            reward_field_for_train = "reward_norm" if args.normalize_rewards else "reward"
            filtered_rollouts, filter_stats = _filter_groups_by_reward_range(
                rollouts,
                reward_field=reward_field_for_train,
                min_range=args.min_group_reward_range,
            )

            if not filtered_rollouts:
                global_step += 1
                print(
                    f"epoch={epoch} step={global_step} skipped=1 "
                    f"reason=no_groups_after_filter filtered_pct={filter_stats['filtered_pct']:.1f}",
                    flush=True,
                )
                continue

            advantages = _compute_group_advantages(filtered_rollouts, reward_field=reward_field_for_train)

            model.train()
            optimizer.zero_grad()
            flat_losses = []
            flat_kls = []

            for chunk_start in range(0, len(filtered_rollouts), args.rollout_micro_batch_size):
                chunk_rollouts = filtered_rollouts[chunk_start : chunk_start + args.rollout_micro_batch_size]
                chunk_advantages = torch.tensor(
                    advantages[chunk_start : chunk_start + args.rollout_micro_batch_size],
                    dtype=torch.float32,
                    device=model.device,
                )
                seq_log_probs = _compute_sequence_logprobs(model, tokenizer.pad_token_id, chunk_rollouts)
                response_lengths = torch.tensor(
                    [max(1, len(rollout["response_token_ids"])) for rollout in chunk_rollouts],
                    dtype=torch.float32,
                    device=model.device,
                )
                policy_norm_log_probs = seq_log_probs / response_lengths
                pg_loss = -(chunk_advantages * policy_norm_log_probs).mean()

                total_loss = pg_loss
                kl_value = 0.0
                if reference_model is not None and args.kl_coef > 0:
                    with torch.no_grad():
                        ref_seq_log_probs = _compute_sequence_logprobs(
                            reference_model,
                            tokenizer.pad_token_id,
                            chunk_rollouts,
                        )
                    ref_norm_log_probs = ref_seq_log_probs / response_lengths
                    # On-policy sampled approximation of KL(pi || pi_ref).
                    kl_per_seq = policy_norm_log_probs - ref_norm_log_probs
                    kl_loss = kl_per_seq.mean()
                    total_loss = total_loss + (args.kl_coef * kl_loss)
                    kl_value = float(kl_loss.detach().cpu().item())

                total_loss.backward()
                flat_losses.append(float(total_loss.detach().cpu()))
                if reference_model is not None and args.kl_coef > 0:
                    flat_kls.append(kl_value)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.eval()

            global_step += 1
            if args.rollout_out_dir:
                rollout_path = Path(args.rollout_out_dir) / f"epoch_{epoch:02d}_step_{global_step:05d}.csv"
                _save_rollouts_csv(filtered_rollouts, rollout_path)

            mean_loss = sum(flat_losses) / max(len(flat_losses), 1)
            mean_kl = sum(flat_kls) / max(len(flat_kls), 1) if flat_kls else 0.0
            metrics = _collect_rollout_metrics(
                filtered_rollouts,
                reward_field=reward_field_for_train,
                mean_kl=mean_kl,
                filtered_pct=filter_stats["filtered_pct"],
            )
            print(
                (
                    f"epoch={epoch} step={global_step} "
                    f"loss={mean_loss:.5f} reward={metrics['mean_reward']:.4f} "
                    f"raw_reward={metrics['mean_raw_reward']:.4f} "
                    f"acc_reward={metrics['mean_answer_correct']:.4f} "
                    f"orm={metrics['mean_orm_score']:.4f} "
                    f"fmt={metrics['mean_format_ok']:.4f} "
                    f"group_var={metrics['mean_group_var']:.4f} "
                    f"filtered_groups={metrics['filtered_groups_pct']:.1f}% "
                    f"avg_len={metrics['avg_response_len']:.1f} "
                    f"kl={metrics['mean_kl']:.5f}"
                ),
                flush=True,
            )

        ckpt_dir = Path(args.out_dir) / f"checkpoint-{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"Saved RL checkpoint -> {ckpt_dir}")

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(Path(args.out_dir) / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    print(f"Saved final RL adapter -> {args.out_dir}")

    del model
    if reference_model is not None:
        del reference_model
    torch.cuda.empty_cache()

    if not args.skip_eval:
        _evaluate_saved_checkpoints(
            args,
            val_prompts,
            val_golds,
            val_prompt_styles,
            quant_config,
            load_4bit,
        )
    else:
        print("Skipping post-training evaluation (--skip-eval).")


def _build_prompt(row: pd.Series, task: str) -> str:
    from run_inference import _build_prompt as build_prompt

    return build_prompt(row, task)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stabilized ORM-based sampling and RL post-training.")
    parser.add_argument("--mode", required=True, choices=["rerank", "score", "train"])
    parser.add_argument("--model-name", required=True, help="Policy model HuggingFace id")
    parser.add_argument("--task", required=True, choices=sorted(SUPPORTED_TASKS))
    parser.add_argument("--orm-backend", default="rlhflow_math_orm", choices=["rlhflow_math_orm", "armo"])
    parser.add_argument("--orm-model-name", default=DEFAULT_ORM_MODEL)
    parser.add_argument("--orm-load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Safer defaults for unstable REINFORCE-style debugging.
    parser.add_argument("--num-samples", type=int, default=8, help="Number of sampled traces per prompt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=2, help="Number of prompts per rollout batch")
    parser.add_argument("--orm-batch-size", type=int, default=4)
    parser.add_argument("--load-in-4bit", action="store_true", help="Force 4-bit quantization for the policy model")

    parser.add_argument("--data-csv", default=None, help="CSV used in score mode")
    parser.add_argument("--split", default="test", help="Score-mode split filter")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out-path", default=None, help="All sampled scored rollouts CSV output path for rerank mode")
    parser.add_argument("--selected-out-path", default=None, help="Selected best-of-N CSV output path for rerank mode")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Judge batch size when computing rerank accuracy")

    parser.add_argument("--train-csv", default=None, help="Training CSV used in train mode")
    parser.add_argument("--out-dir", default=None, help="Directory for RL checkpoints and final adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--rollout-micro-batch-size", type=int, default=8)
    parser.add_argument("--reward-field", choices=["reward", "orm_score", "reward_norm"], default="reward")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--rollout-out-dir", default=None, help="Optional directory to save per-step rollout CSVs during RL")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--skip-eval", action="store_true")

    # Combined anchored reward.
    parser.add_argument("--w-answer-correct", type=float, default=1.0)
    parser.add_argument("--w-format", type=float, default=0.1)
    parser.add_argument("--w-orm", type=float, default=0.1)

    # Reward stabilization.
    parser.add_argument("--reward-clip-min", type=float, default=None)
    parser.add_argument("--reward-clip-max", type=float, default=None)
    parser.add_argument("--normalize-rewards", action="store_true")
    parser.add_argument("--min-group-reward-range", type=float, default=0.05)

    # KL regularization to frozen reference policy.
    parser.add_argument("--kl-coef", type=float, default=0.0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _set_seed(args.seed)

    if args.orm_backend == "armo" and args.orm_model_name == DEFAULT_ORM_MODEL:
        args.orm_model_name = DEFAULT_ARMO_MODEL

    if args.mode in RERANK_ALIASES:
        if not args.data_csv or not args.out_path:
            parser.error("rerank mode requires --data-csv and --out-path")
        _finalize_output_args(args)
        if args.mode == "score":
            print("mode=score is currently an alias for mode=rerank")
        if not args.selected_out_path:
            out_path = Path(args.out_path)
            args.selected_out_path = str(out_path.with_name(out_path.stem + "_selected.csv"))
        _require_scratch_path(args.out_path, "--out-path")
        _require_scratch_path(args.selected_out_path, "--selected-out-path")
        print(f"Resolved rerank outputs: out_path={args.out_path} selected_out_path={args.selected_out_path}")
        rerank_with_orm(args)
        return

    if not args.train_csv or not args.out_dir:
        parser.error("train mode requires --train-csv and --out-dir")
    if args.num_samples < 2:
        parser.error("train mode requires --num-samples >= 2")

    _require_scratch_path(args.out_dir, "--out-dir")
    _require_scratch_path(args.rollout_out_dir, "--rollout-out-dir")
    _finalize_output_args(args)
    print(f"Resolved training output dir: out_dir={args.out_dir}")
    if args.rollout_out_dir:
        print(f"Resolved rollout output dir: rollout_out_dir={args.rollout_out_dir}")

    train_with_orm(args)


if __name__ == "__main__":
    main()
