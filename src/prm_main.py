#!/usr/bin/env python3
"""PRM-based sampling and RL post-training.

This script covers 2 experimental paths with a process reward model (PRM):

1) `mode=rerank`
     No-training use of a PRM. Sample N reasoning traces per test prompt, score
     each trace with a PRM, select the best sample per prompt, and report final
     accuracy for the selected samples.

2) `mode=train`
   Run a lightweight RL post-training loop using PRM rewards. The update is a
   group-normalized REINFORCE objective over sampled traces with LoRA adapters,
   then evaluate saved checkpoints with the existing repo evaluation logic.

Two PRM backends are supported:
    - `qwen_math`     : Qwen/Qwen2.5-Math-PRM-7B step-classifier PRM.
    - `rlhflow_mathrm`: RLHFlow Llama3.1-8B math-rm PRMs (Deepseek/Mistral data),
                        decoded ThinkPRM-style (each step -> user turn + assistant
                        "+"/"-" label, P("+") per step via label-position masking).

Important limitation:
    - For now, PRM use is enabled only for `correct_answer`.
    - `distractor` is intentionally deferred because the reward definition you
        want is misconception-faithfulness, which likely needs a specialized PRM.
    - `next_subquestion` is intentionally unsupported because the current
        pipeline generates a single subquestion rather than a step-by-step trace.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_inference import (  # noqa: E402
    TASK_MAX_NEW_TOKENS,
    _build_prompt,
    _needs_4bit,
    score_correct_answer,
    generate_predictions,
)

DEFAULT_PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
DEFAULT_PRM_SYSTEM_PROMPT = "Please reason step by step, and put your final answer at the end."
SUPPORTED_TASKS = {"correct_answer"}
STEP_SEPARATOR = "<extra_0>"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
RERANK_ALIASES = {"rerank", "score"}


if not hasattr(DynamicCache, "from_legacy_cache"):
    @classmethod
    def _from_legacy_cache(cls, past_key_values):
        cache = cls()
        if past_key_values is None:
            return cache
        for layer_idx, layer_past in enumerate(past_key_values):
            if not isinstance(layer_past, (tuple, list)) or len(layer_past) < 2:
                continue
            key_states, value_states = layer_past[0], layer_past[1]
            cache.update(key_states, value_states, layer_idx)
        return cache

    DynamicCache.from_legacy_cache = _from_legacy_cache


if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        del new_seq_length
        return self.get_seq_length(layer_idx)

    DynamicCache.get_usable_length = _get_usable_length


if not hasattr(DynamicCache, "to_legacy_cache"):
    def _to_legacy_cache(self):
        legacy_cache = []
        for layer in self.layers:
            legacy_cache.append((layer.keys, layer.values))
        return tuple(legacy_cache)

    DynamicCache.to_legacy_cache = _to_legacy_cache


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_policy_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _make_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _load_policy_model(model_name: str, load_in_4bit: bool, lora_r: int, lora_alpha: int):
    from peft import LoraConfig, TaskType, get_peft_model

    quant_config = _make_quant_config(load_in_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model, quant_config


def _format_generation_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _trim_generated_ids(token_ids: list[int], pad_token_id: int | None, eos_token_id: int | None) -> list[int]:
    trimmed = []
    for token_id in token_ids:
        if pad_token_id is not None and token_id == pad_token_id:
            break
        trimmed.append(int(token_id))
        if eos_token_id is not None and token_id == eos_token_id:
            break
    return trimmed


def _split_reasoning_steps(response: str) -> list[str]:
    steps = [part.strip() for part in re.split(r"\n\s*\n+", str(response).strip()) if part.strip()]
    return steps or [str(response).strip()]


def _aggregate_prm_step_scores(step_scores: list[float]) -> dict[str, Any]:
    """Aggregate per-step P(+) scores into the shared PRM reward dictionary."""
    reward_product = 1.0
    reward_logsum = 0.0
    reward_sum = 0.0
    for value in step_scores:
        clipped = max(min(float(value), 1.0), 1e-12)
        reward_product *= clipped
        reward_logsum += math.log(clipped)
        reward_sum += clipped
    num_steps = len(step_scores)
    reward_mean_log = reward_logsum / max(num_steps, 1)
    reward_mean = reward_sum / max(num_steps, 1)
    return {
        "reward_mean_log": reward_mean_log,
        "reward_mean": reward_mean,
        "step_scores": step_scores,
        "reward_product": reward_product,
        "reward_logsum": reward_logsum,
        "num_steps": num_steps,
    }


class BaseProcessRewardModel:
    """Minimal PRM interface so future custom-trained PRMs can plug in here."""

    def score_responses(
        self,
        prompts: list[str],
        responses: list[str],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class QwenMathProcessRewardModel(BaseProcessRewardModel):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        load_in_4bit: bool,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.load_in_4bit = load_in_4bit
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if getattr(self.config, "pad_token_id", None) is None:
            self.config.pad_token_id = self.tokenizer.pad_token_id
        quant_config = _make_quant_config(load_in_4bit)
        self.model = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        sep_ids = self.tokenizer.encode(STEP_SEPARATOR, add_special_tokens=False)
        if len(sep_ids) != 1:
            raise ValueError(f"{STEP_SEPARATOR} is not a single token: {sep_ids}")
        self.step_sep_id = sep_ids[0]

    def _build_conversation(self, prompt: str, response: str) -> str:
        steps = _split_reasoning_steps(response)
        assistant_content = STEP_SEPARATOR.join(steps) + STEP_SEPARATOR
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def score_responses(
        self,
        prompts: list[str],
        responses: list[str],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        conversations = [
            self._build_conversation(prompt, response)
            for prompt, response in zip(prompts, responses)
        ]
        all_results = []
        for start in tqdm(range(0, len(conversations), batch_size), desc="PRM scoring"):
            batch_texts = conversations[start : start + batch_size]
            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**enc)
            logits = outputs[0]
            token_masks = enc["input_ids"] == self.step_sep_id
            for sample_idx in range(logits.size(0)):
                step_logits = logits[sample_idx, token_masks[sample_idx], :]
                if step_logits.numel() == 0:
                    step_scores = []
                else:
                    step_probs = F.softmax(step_logits.float(), dim=-1)
                    if step_probs.shape[-1] != 2:
                        raise ValueError(
                            f"Expected PRM logits with last dimension 2, got {tuple(step_probs.shape)}"
                        )
                    step_scores = step_probs[:, 1].detach().cpu().tolist()
                all_results.append(_aggregate_prm_step_scores(step_scores))
        return all_results


class RLHFlowProcessRewardModel(BaseProcessRewardModel):
    """RLHFlow math-rm PRM using ThinkPRM-style label-masking decoding.

    Each reasoning step becomes a user turn followed by an assistant "+"/"-"
    label token. The probability the model assigns to "+" at each labeled
    position is used as that step's process reward, then aggregated across
    steps. Assistant-label positions are located with a parallel conversation
    in which every "+" is swapped for the dummy token "ки".
    """

    def __init__(self, model_name: str, load_in_4bit: bool) -> None:
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
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
                f"RLHFlow PRM expects single-token +/- labels, got {plus_tag_id} and {minus_tag_id}"
            )
        self.candidate_tokens = [plus_tag_id[0], minus_tag_id[0]]
        # Dummy token used to locate assistant-label positions (ThinkPRM masking trick).
        self.special_tok_id = int(self.tokenizer("ки", return_tensors="pt").input_ids[0, 1])

    def _build_conversations(
        self, prompt: str, response: str
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        steps = _split_reasoning_steps(response)
        conversation: list[dict[str, str]] = []
        conversation_mask: list[dict[str, str]] = []
        for step_idx, step in enumerate(steps):
            if step_idx == 0:
                text = f"{prompt.strip()} {step.strip()}".strip()
            else:
                text = step.strip()
            conversation.append({"content": text, "role": "user"})
            conversation.append({"content": "+", "role": "assistant"})
            conversation_mask.append({"content": text, "role": "user"})
            conversation_mask.append({"content": "ки", "role": "assistant"})
        return conversation, conversation_mask

    def score_responses(
        self,
        prompts: list[str],
        responses: list[str],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        conversation_pairs = [
            self._build_conversations(prompt, response)
            for prompt, response in zip(prompts, responses)
        ]
        all_results: list[dict[str, Any]] = []
        for start in tqdm(range(0, len(conversation_pairs), batch_size), desc="PRM scoring"):
            batch_pairs = conversation_pairs[start : start + batch_size]
            conversations = [pair[0] for pair in batch_pairs]
            conversations_mask = [pair[1] for pair in batch_pairs]

            inputs = self.tokenizer.apply_chat_template(
                conversations,
                padding=True,
                return_tensors="pt",
                return_dict=False,
            ).to(self.model.device)
            inputs_mask = self.tokenizer.apply_chat_template(
                conversations_mask,
                padding=True,
                return_tensors="pt",
                return_dict=False,
            ).to(self.model.device)
            if inputs.shape != inputs_mask.shape:
                raise ValueError(
                    f"RLHFlow mask conversation shape {tuple(inputs_mask.shape)} does not match "
                    f"label conversation shape {tuple(inputs.shape)}"
                )

            with torch.no_grad():
                # logits at position t predict token t+1, so the assistant-label
                # probability lives at the position before the dummy/"+" token.
                logits = self.model(inputs).logits[:, :, self.candidate_tokens]
                probs_plus = logits.softmax(dim=-1)[:, :, 0]

            for row_idx in range(len(conversations)):
                label_mask = inputs_mask[row_idx, 1:] == self.special_tok_id
                step_scores = (
                    probs_plus[row_idx, :-1][label_mask].float().detach().cpu().tolist()
                )
                all_results.append(_aggregate_prm_step_scores(step_scores))
        return all_results


def build_process_reward_model(args) -> BaseProcessRewardModel:
    if args.prm_backend == "qwen_math":
        return QwenMathProcessRewardModel(
            args.prm_model_name,
            system_prompt=args.prm_system_prompt,
            load_in_4bit=args.prm_load_in_4bit,
        )
    if args.prm_backend == "rlhflow_mathrm":
        return RLHFlowProcessRewardModel(
            args.prm_model_name,
            load_in_4bit=args.prm_load_in_4bit,
        )
    raise ValueError(f"Unsupported --prm-backend: {args.prm_backend}")


def _load_dataframe(train_csv: str, task: str, split: str | None = None, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split].copy()
    if task == "correct_answer" and "reasoning_trace" in df.columns and "target_question_reasoning_trace" in df.columns:
        mask = df["target_question_reasoning_trace"].isna() | (df["target_question_reasoning_trace"].astype(str).str.strip() == "")
        df.loc[mask, "target_question_reasoning_trace"] = df.loc[mask, "reasoning_trace"]
    if max_rows is not None:
        df = df.head(max_rows).copy()
    return df.reset_index(drop=True)


def _row_metadata(row: pd.Series) -> dict[str, Any]:
    metadata = {
        "question_id": row.get("question_id"),
        "split": row.get("split"),
        "prompt_style": row.get("prompt_style"),
    }
    if "pair_index" in row:
        metadata["pair_index"] = row.get("pair_index")
    return metadata


def _sample_rollouts(
    model,
    tokenizer,
    prompts: list[str],
    metadata: list[dict[str, Any]],
    task: str,
    num_samples: int,
    batch_size: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    model.eval()
    flat_rollouts: list[dict[str, Any]] = []
    max_new_tokens = TASK_MAX_NEW_TOKENS[task]

    for start in tqdm(range(0, len(prompts), batch_size), desc="Sampling rollouts"):
        batch_prompts = prompts[start : start + batch_size]
        batch_meta = metadata[start : start + batch_size]
        formatted = [_format_generation_prompt(tokenizer, prompt) for prompt in batch_prompts]
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for batch_index, prompt in enumerate(batch_prompts):
            meta = batch_meta[batch_index]
            prompt_ids = enc["input_ids"][batch_index][enc["attention_mask"][batch_index].bool()].tolist()
            for sample_index in range(num_samples):
                seq_index = batch_index * num_samples + sample_index
                response_ids = _trim_generated_ids(
                    generated[seq_index][input_len:].tolist(),
                    tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
                )
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                flat_rollouts.append(
                    {
                        "group_id": start + batch_index,
                        "sample_index": sample_index,
                        "prompt": prompt,
                        "response": response_text,
                        "prompt_token_ids": prompt_ids,
                        "response_token_ids": response_ids,
                        **meta,
                    }
                )

    return flat_rollouts


def _compute_sequence_logprobs(model, pad_token_id: int, rollout_batch: list[dict[str, Any]]) -> torch.Tensor:
    input_tensors = []
    label_tensors = []
    for rollout in rollout_batch:
        prompt_ids = rollout["prompt_token_ids"]
        response_ids = rollout["response_token_ids"]
        full_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
        labels = torch.tensor(([-100] * len(prompt_ids)) + response_ids, dtype=torch.long)
        input_tensors.append(full_ids)
        label_tensors.append(labels)

    batch_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_tensors,
        batch_first=True,
        padding_value=pad_token_id,
    ).to(model.device)
    batch_labels = torch.nn.utils.rnn.pad_sequence(
        label_tensors,
        batch_first=True,
        padding_value=-100,
    ).to(model.device)
    attention_mask = (batch_input_ids != pad_token_id).long()

    outputs = model(input_ids=batch_input_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = batch_labels[:, 1:]
    active_mask = shift_labels != -100

    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.masked_fill(~active_mask, 0)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    seq_log_probs = (token_log_probs * active_mask).sum(dim=-1)
    return seq_log_probs


def _attach_prm_scores(
    prm: BaseProcessRewardModel,
    rollouts: list[dict[str, Any]],
    batch_size: int,
) -> None:
    results = prm.score_responses(
        [rollout["prompt"] for rollout in rollouts],
        [rollout["response"] for rollout in rollouts],
        batch_size=batch_size,
    )
    for rollout, prm_result in zip(rollouts, results):
        rollout.update(prm_result)


def _select_best_rollouts(rollouts: list[dict[str, Any]], reward_field: str) -> list[dict[str, Any]]:
    best_by_group: dict[int, dict[str, Any]] = {}
    for rollout in rollouts:
        group_id = int(rollout["group_id"])
        current_best = best_by_group.get(group_id)
        if current_best is None or float(rollout[reward_field]) > float(current_best[reward_field]):
            best_by_group[group_id] = rollout
    return [best_by_group[group_id] for group_id in sorted(best_by_group)]


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
                "selected_reward_product": best_rollout.get("reward_product"),
                "selected_reward_logsum": best_rollout.get("reward_logsum"),
                "selected_reward_mean_log": best_rollout.get("reward_mean_log"),
                "selected_reward_mean": best_rollout.get("reward_mean"),
                "selected_num_steps": best_rollout.get("num_steps"),
                "selected_step_scores": json.dumps(best_rollout.get("step_scores", [])),
                "score": score,
                "correct": "Yes" if score else "No",
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _build_val_data(df: pd.DataFrame, task: str):
    prompts, golds, prompt_styles = [], [], []
    for _, row in df.iterrows():
        prompts.append(_build_prompt(row, task))
        prompt_styles.append(str(row.get("prompt_style", "")))
        if task == "correct_answer":
            golds.append(str(row["target_answer"]))
        else:
            raise ValueError(f"Unsupported task for PRM validation: {task}")
    return prompts, golds, prompt_styles


def _evaluate_saved_checkpoints(
    args,
    val_prompts: list[str],
    val_golds: list[str],
    val_prompt_styles: list[str],
    quant_config,
    load_4bit: bool,
) -> None:
    ckpt_dirs = sorted(Path(args.out_dir).glob("checkpoint-*"))
    if not ckpt_dirs:
        print("No checkpoints found to evaluate.")
        return

    print(f"\n{'=' * 60}")
    print(f"Post-training evaluation: {len(ckpt_dirs)} checkpoints, {len(val_prompts)} val prompts")
    print(f"{'=' * 60}")

    tokenizer = _load_policy_tokenizer(args.model_name)
    best_acc = -1.0
    best_ckpt = None

    for ckpt_dir in ckpt_dirs:
        print(f"\nGenerating predictions for {ckpt_dir.name}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if not load_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        )
        from peft import PeftModel  # local import to keep top-level lighter

        model = PeftModel.from_pretrained(model, str(ckpt_dir))
        model = model.merge_and_unload()
        model.eval()

        predictions = generate_predictions(
            model,
            tokenizer,
            val_prompts,
            TASK_MAX_NEW_TOKENS[args.task],
            batch_size=args.eval_batch_size,
        )
        del model
        torch.cuda.empty_cache()

        scores = score_correct_answer(
            predictions,
            val_golds,
            val_prompts,
            val_prompt_styles,
            batch_size=args.eval_batch_size,
        )
        accuracy = sum(scores) / len(scores) if scores else 0.0
        print(f"  {ckpt_dir.name}: accuracy = {accuracy:.4f} ({sum(scores)}/{len(scores)})")

        with open(ckpt_dir / "val_accuracy.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "checkpoint": ckpt_dir.name,
                    "accuracy": accuracy,
                    "n_correct": sum(scores),
                    "n_samples": len(scores),
                },
                handle,
                indent=2,
            )

        if accuracy > best_acc:
            best_acc = accuracy
            best_ckpt = str(ckpt_dir)

    if best_ckpt is not None:
        print(f"\nBest checkpoint: {best_ckpt} (accuracy={best_acc:.4f})")
        with open(Path(args.out_dir) / "best_checkpoint.json", "w", encoding="utf-8") as handle:
            json.dump({"path": best_ckpt, "accuracy": best_acc}, handle, indent=2)


def _compute_group_advantages(rollouts: list[dict[str, Any]], reward_field: str) -> list[float]:
    advantages = [0.0] * len(rollouts)
    grouped_indices: dict[int, list[int]] = {}
    for idx, rollout in enumerate(rollouts):
        grouped_indices.setdefault(int(rollout["group_id"]), []).append(idx)

    for indices in grouped_indices.values():
        rewards = torch.tensor([float(rollouts[idx][reward_field]) for idx in indices], dtype=torch.float32)
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        if torch.isnan(std) or std.item() < 1e-6:
            normalized = rewards - mean
        else:
            normalized = (rewards - mean) / (std + 1e-6)
        for idx, advantage in zip(indices, normalized.tolist()):
            advantages[idx] = float(advantage)
    return advantages


def _save_rollouts_csv(rollouts: list[dict[str, Any]], out_path: Path) -> None:
    rows = []
    for rollout in rollouts:
        row = {key: value for key, value in rollout.items() if key not in {"prompt_token_ids", "response_token_ids"}}
        row["step_scores"] = json.dumps(rollout.get("step_scores", []))
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def rerank_with_prm(args) -> None:
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(
            f"PRM scoring currently supports only {sorted(SUPPORTED_TASKS)}; got task={args.task}."
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

    prm = build_process_reward_model(args)
    _attach_prm_scores(prm, rollouts, batch_size=args.prm_batch_size)

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
    print(f"Saved PRM-selected best-of-N responses to {selected_out_path}")
    print(f"Best-of-{args.num_samples} accuracy: {accuracy:.1%} ({sum(scores)}/{len(scores)})")


def train_with_prm(args) -> None:
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(
            f"PRM RL currently supports only {sorted(SUPPORTED_TASKS)}; got task={args.task}."
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

    prm = build_process_reward_model(args)
    val_prompts, val_golds, val_prompt_styles = _build_val_data(val_df, task=args.task)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_df = train_df.sample(frac=1, random_state=args.seed + epoch).reset_index(drop=True)
        progress = tqdm(range(0, len(epoch_df), args.batch_size), desc=f"RL epoch {epoch}")

        for start in progress:
            batch_df = epoch_df.iloc[start : start + args.batch_size].reset_index(drop=True)
            prompts = [_build_prompt(row, args.task) for _, row in batch_df.iterrows()]
            metadata = [_row_metadata(row) for _, row in batch_df.iterrows()]

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
            _attach_prm_scores(prm, rollouts, batch_size=args.prm_batch_size)
            advantages = _compute_group_advantages(rollouts, reward_field=args.reward_field)

            model.train()
            optimizer.zero_grad()
            flat_losses = []
            for chunk_start in range(0, len(rollouts), args.rollout_micro_batch_size):
                chunk_rollouts = rollouts[chunk_start : chunk_start + args.rollout_micro_batch_size]
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
                normalized_log_probs = seq_log_probs / response_lengths
                loss = -(chunk_advantages * normalized_log_probs).mean()
                loss.backward()
                flat_losses.append(float(loss.detach().cpu()))

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.eval()

            mean_reward = sum(float(rollout[args.reward_field]) for rollout in rollouts) / max(len(rollouts), 1)
            progress.set_postfix(
                reward=f"{mean_reward:.4g}",
                loss=f"{sum(flat_losses) / max(len(flat_losses), 1):.4f}",
            )
            global_step += 1

            if args.rollout_out_dir:
                rollout_path = Path(args.rollout_out_dir) / f"epoch_{epoch:02d}_step_{global_step:05d}.csv"
                _save_rollouts_csv(rollouts, rollout_path)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRM-based sampling and RL post-training.")
    parser.add_argument("--mode", required=True, choices=["rerank", "score", "train"])
    parser.add_argument("--model-name", required=True, help="Policy model HuggingFace id")
    parser.add_argument("--task", required=True, choices=sorted(SUPPORTED_TASKS))
    parser.add_argument("--prm-backend", default="qwen_math", choices=["qwen_math", "rlhflow_mathrm"])
    parser.add_argument("--prm-model-name", default=DEFAULT_PRM_MODEL)
    parser.add_argument("--prm-system-prompt", default=DEFAULT_PRM_SYSTEM_PROMPT)
    parser.add_argument("--prm-load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=4, help="Number of sampled traces per prompt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=2, help="Number of prompts per rollout batch")
    parser.add_argument("--prm-batch-size", type=int, default=4)
    parser.add_argument("--load-in-4bit", action="store_true", help="Force 4-bit quantization for the policy model")

    parser.add_argument("--data-csv", default=None, help="CSV used in score mode")
    parser.add_argument("--split", default="test", help="Score-mode split filter")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out-path", default=None, help="All sampled scored rollouts CSV output path for rerank mode")
    parser.add_argument("--selected-out-path", default=None, help="Selected best-of-N CSV output path for rerank mode")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Judge batch size when computing rerank accuracy")

    parser.add_argument("--train-csv", default=None, help="Training CSV used in train mode")
    parser.add_argument("--out-dir", default=None, help="Directory for RL checkpoints and final adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rollout-micro-batch-size", type=int, default=8)
    parser.add_argument(
        "--reward-field",
        choices=["reward_product", "reward_logsum", "reward_mean_log", "reward_mean"],
        default="reward_mean_log",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rollout-out-dir", default=None, help="Optional directory to save per-step rollout CSVs during RL")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--skip-eval", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _set_seed(args.seed)

    if args.mode in RERANK_ALIASES:
        if not args.data_csv or not args.out_path:
            parser.error("rerank mode requires --data-csv and --out-path")
        if args.mode == "score":
            print("mode=score is currently an alias for mode=rerank")
        if not args.selected_out_path:
            out_path = Path(args.out_path)
            args.selected_out_path = str(out_path.with_name(out_path.stem + "_selected.csv"))
        rerank_with_prm(args)
        return

    if not args.train_csv or not args.out_dir:
        parser.error("train mode requires --train-csv and --out-dir")
    if args.num_samples < 2:
        parser.error("train mode requires --num-samples >= 2")
    train_with_prm(args)


if __name__ == "__main__":
    main()
