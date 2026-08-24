#!/usr/bin/env python3
"""Small ORM (outcome reward model) smoke test on local math datasets.

This mirrors ``test_prm_samples.py`` but for *outcome* reward models, which score
a whole (prompt, response) pair with a single scalar instead of producing one
score per reasoning step. Use it to inspect ORM behaviour (e.g. does a correct
math trace score higher than a wrong one?) before wiring an ORM into the
best-of-N / RL pipeline.

Four decoding backends are supported, each copied from the official model card
so the decode path is guaranteed correct:

  - ``seqcls``   : sequence-classifier reward models, score = ``rm(ids).logits[0][0]``.
                   Official cards: Skywork-Reward-Llama-3.1-8B-v0.2, GRM-Llama3-8B.
    - ``internlm2``: InternLM2 reward custom head, score = ``model.get_score(tok, chat)``.
    - ``rlhflow_orm``: RLHFlow math ORM, score = $P(+)$ from the assistant sign token
                                         after formatting ``[{user: prompt + response}, {assistant: '+'}]``.
  - ``eurus``    : Eurus-RM-7b custom head, score = ``model(**inputs).item()`` with
                   a Mistral ``[INST] ... [/INST] ...`` template.

All catalogued ORMs are math/reasoning-capable and <= 10B parameters so they fit
a single 24 GB GPU (4-bit optional via ``--orm-load-in-4bit``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from prm_main import (
    DEFAULT_PRM_SYSTEM_PROMPT,
    _format_generation_prompt,
    _load_policy_tokenizer,
    _make_quant_config,
    _split_reasoning_steps,
    _trim_generated_ids,
)
from run_inference import _build_prompt, _needs_4bit, TASK_MAX_NEW_TOKENS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = {
    "eedi": ROOT / "eedi/data/processed/correct_answer_pairs_eedi.csv",
    "gsm8k": ROOT / "reasoning-efficiency/experiments/proof_search/out/correct_answer_pairs_gsm8k.csv",
}
DEFAULT_POLICY_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
POLICY_PRESETS = {
    "qwen_instruct_sweep": DEFAULT_POLICY_MODELS,
}

# ── ORM catalog (all math/reasoning-capable, <= 10B) ──────────────────────────
ORM_PRESETS = {
    "eurus_rm_7b": {
        "label": "Eurus-RM-7B",
        "backend": "eurus",
        "model_name": "openbmb/Eurus-RM-7b",
        "supported": True,
        "notes": (
            "Reasoning/math-specialised RM (UltraInteract). Mistral [INST] template, "
            "scalar reward = model(**inputs).item(). Higher is better."
        ),
    },
    "internlm2_reward_7b": {
        "label": "InternLM2-7B-Reward",
        "backend": "internlm2",
        "model_name": "internlm/internlm2-7b-reward",
        "supported": True,
        "notes": (
            "Custom reward head trained on 2.4M prefs incl. math. "
            "score = model.get_score(tokenizer, chat). Higher is better."
        ),
    },
    "internlm2_reward_1_8b": {
        "label": "InternLM2-1.8B-Reward",
        "backend": "internlm2",
        "model_name": "internlm/internlm2-1_8b-reward",
        "supported": True,
        "notes": "Small (1.8B) InternLM2 reward head; cheapest default for a 24 GB GPU.",
    },
    "skywork_reward_llama_8b": {
        "label": "Skywork-Reward-Llama-3.1-8B-v0.2",
        "backend": "seqcls",
        "model_name": "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        "supported": True,
        "notes": (
            "Bradley-Terry sequence classifier (num_labels=1). "
            "score = rm(apply_chat_template(conv)).logits[0][0]. Strong on math (RewardBench)."
        ),
    },
    "grm_llama3_8b": {
        "label": "GRM-Llama3-8B-rewardmodel-ft",
        "backend": "seqcls",
        "model_name": "Ray2333/GRM-Llama3-8B-rewardmodel-ft",
        "supported": True,
        "notes": "Generalisable RM, same sequence-classifier decode path as Skywork.",
    },
    "rlhflow_orm_mistral_8b": {
        "label": "RLHFlow Llama3.1-8B-ORM-Mistral-Data",
        "backend": "rlhflow_orm",
        "model_name": "RLHFlow/Llama3.1-8B-ORM-Mistral-Data",
        "supported": True,
        "notes": (
            "Direct ORM counterpart to RLHFlow/Llama3.1-8B-PRM-Mistral-Data. "
            "Published eval scores P('+') from a single-turn chat with user=question+trace."
        ),
    },
    "rlhflow_orm_deepseek_8b": {
        "label": "RLHFlow Llama3.1-8B-ORM-Deepseek-Data",
        "backend": "rlhflow_orm",
        "model_name": "RLHFlow/Llama3.1-8B-ORM-Deepseek-Data",
        "supported": True,
        "notes": (
            "Direct ORM counterpart to RLHFlow/Llama3.1-8B-PRM-Deepseek-Data. "
            "Published eval scores P('+') from a single-turn chat with user=question+trace."
        ),
    },
}

META_STEP_PATTERNS: list[str] = []

# ── curated GSM8K / EEDI correct-vs-wrong traces (shared with PRM smoke test) ──
CURATED_TRACE_CASES = [
    {
        "dataset": "gsm8k",
        "case_id": "gsm8k_1_natalia_clips",
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "correct_trace": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.",
        "wrong_trace": "Natalia sold half as many clips in May, so she sold 48 * 2 = 96 clips in May.\nAltogether she sold 48 + 96 = 144 clips.",
    },
    {
        "dataset": "gsm8k",
        "case_id": "gsm8k_2_weng_babysitting",
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "correct_trace": "Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $10.",
        "wrong_trace": "Weng earns $12 in 60 minutes.\nIn 50 minutes, she earns 12 * 50 = $600.",
    },
    {
        "dataset": "gsm8k",
        "case_id": "gsm8k_3_betty_wallet",
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "correct_trace": "In the beginning, Betty has only 100 / 2 = $50.\nBetty's grandparents gave her 15 * 2 = $30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $5 more.",
        "wrong_trace": "Betty starts with $50.\nHer parents and grandparents give her 15 + 30 = $45 more, so now she has $95.\nShe still needs 100 - 95 = $15.",
    },
    {
        "dataset": "eedi",
        "case_id": "eedi_1_brackets_13",
        "question": "\\[\n3 \\times 2+4-5\n\\]\nWhere do the brackets need to go to make the answer equal \\( 13 \\) ?",
        "correct_trace": "Without brackets: 3*2+4-5 = 6+4-5 = 5, which is not 13.\nIf we bracket (2+4): 3*(2+4)-5 = 3*6-5 = 18-5 = 13.\nSo the brackets go as 3 x (2+4) - 5.",
        "wrong_trace": "Put brackets around 4-5 first.\nThen 3 times 2 plus (4-5) becomes 6 + (-1) = 13.\nSo the answer is 3 times 2 plus (4-5).",
    },
    {
        "dataset": "eedi",
        "case_id": "eedi_2_does_not_simplify",
        "question": "Simplify the following, if possible: \\( \\frac{m^{2}+2 m-3}{m-3} \\)",
        "correct_trace": "Factor the numerator: m^2+2m-3 = (m+3)(m-1).\nThe denominator m-3 shares no common factor with the numerator.\nTherefore the expression does not simplify.",
        "wrong_trace": "Factor the numerator as (m+3)(m-1).\nCancel the m-3 in the denominator with the m-1 in the numerator.\nSo the expression simplifies to m+3.",
    },
    {
        "dataset": "eedi",
        "case_id": "eedi_5_percent_fraction",
        "question": "Convert this percentage to a fraction\n\\( 62 \\% \\)",
        "correct_trace": "A percent means out of 100, so 62% = 62/100.\nDivide numerator and denominator by 2: 62/100 = 31/50.\nSo the answer is 31/50.",
        "wrong_trace": "62% means 62 out of 10, so the fraction is 62/10.\nSimplifying gives 31/5.",
    },
]

INLINE_PROBLEM = "Solve for x: 2^(x+1) + 2^x = 24."
INLINE_SOLUTIONS = [
    # Correct
    "\n\n".join(
        [
            "We use the identity 2^(x+1) = 2 * 2^x.",
            "Therefore, 2^(x+1) + 2^x = 2 * 2^x + 2^x = 3 * 2^x.",
            "So the equation becomes 3 * 2^x = 24.",
            "Dividing both sides by 3 gives 2^x = 8.",
            "Since 8 = 2^3, we get x = 3.",
        ]
    ),
    # Wrong: invalid exponent manipulation
    "\n\n".join(
        [
            "We combine the two powers by adding their exponents.",
            "Thus 2^(x+1) + 2^x = 2^((x+1)+x) = 2^(2x+1).",
            "So the equation becomes 2^(2x+1) = 24.",
            "Taking log base 2 gives 2x + 1 = log_2(24).",
            "Therefore x = (log_2(24) - 1) / 2.",
        ]
    ),
    # Very wrong
    "\n\n".join(
        [
            "We subtract the exponents and get 2^(x+1) + 2^x = 2.",
            "So the equation becomes 2 = 24.",
            "Since this is false, there is no solution.",
        ]
    ),
]


# ── mode helpers ──────────────────────────────────────────────────────────────
def _use_inline_solution_mode(args: argparse.Namespace) -> bool:
    return args.score_inline_solutions or args.use_inline_solutions


def _use_curated_trace_mode(args: argparse.Namespace) -> bool:
    return args.score_curated_traces


def _get_curated_trace_cases(dataset: str) -> list[dict[str, str]]:
    if dataset == "both":
        return CURATED_TRACE_CASES
    return [case for case in CURATED_TRACE_CASES if case["dataset"] == dataset]


def _filter_curated_cases_by_id(
    curated_cases: list[dict[str, str]],
    case_id: str | None,
) -> list[dict[str, str]]:
    if case_id is None:
        return curated_cases
    filtered = [case for case in curated_cases if case["case_id"] == case_id]
    if not filtered:
        raise ValueError(f"No curated trace case found for --curated-case-id {case_id!r}")
    return filtered


def _normalize_curated_trace_text(trace: str) -> str:
    lines = [line.strip() for line in str(trace).splitlines() if line.strip()]
    return "\n\n".join(lines)


def _expand_policy_models(policy_models: list[str] | None, policy_preset: str | None) -> list[str]:
    models: list[str] = []
    if policy_preset:
        models.extend(POLICY_PRESETS[policy_preset])
    if policy_models:
        models.extend(policy_models)
    if not models:
        models.append("Qwen/Qwen2.5-1.5B-Instruct")
    return list(dict.fromkeys(models))


def _resolve_orm_specs(
    orm_model_names: list[str] | None,
    orm_presets: list[str] | None,
    default_backend: str,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for preset_name in orm_presets or []:
        preset = ORM_PRESETS[preset_name]
        specs.append(dict(preset))
    for model_name in orm_model_names or []:
        specs.append(
            {
                "label": model_name,
                "backend": default_backend,
                "model_name": model_name,
                "supported": True,
                "notes": f"Custom ORM passed on the CLI (decoded with backend={default_backend}).",
            }
        )
    if not specs:
        default_preset = ORM_PRESETS["internlm2_reward_1_8b"]
        specs.append(dict(default_preset))

    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for spec in specs:
        key = (str(spec["backend"]), str(spec["model_name"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


def _print_available_options() -> None:
    print("Policy presets:")
    for name, models in POLICY_PRESETS.items():
        print(f"  {name}: {', '.join(models)}")
    print()
    print("ORM presets (all <= 10B, math/reasoning-capable):")
    for name, spec in ORM_PRESETS.items():
        print(f"  {name}: backend={spec['backend']} model={spec['model_name']}")
        print(f"    {spec['notes']}")


def _load_sanitized_config(model_name: str):
    """Load a config and fix the InternLM2 ``rope_scaling`` schema mismatch.

    InternLM2's bundled modeling code only understands ``rope_scaling`` being
    ``None`` or an old-style ``{"type": "linear"|"dynamic", "factor": ...}`` dict.
    Newer transformers writes ``{"rope_type": "default", ...}`` for plain RoPE,
    which makes the custom code raise ``KeyError: 'type'`` / ``'factor'``. We map
    a genuine linear/dynamic scaling onto the old schema and otherwise disable
    ``rope_scaling`` (the default-RoPE case) so the model loads.
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
        factor = rope_scaling.get("factor")
        if rope_type in ("linear", "dynamic") and factor is not None:
            config.rope_scaling = {"type": rope_type, "factor": factor}
        else:
            config.rope_scaling = None
    return config


def _expand_embedding_rows(embedding: nn.Embedding, new_num_embeddings: int) -> nn.Embedding:
    old_weight = embedding.weight.data
    old_num_embeddings, embedding_dim = old_weight.shape
    if new_num_embeddings <= old_num_embeddings:
        return embedding

    expanded = nn.Embedding(
        new_num_embeddings,
        embedding_dim,
        padding_idx=embedding.padding_idx,
        device=old_weight.device,
        dtype=old_weight.dtype,
    )
    with torch.no_grad():
        expanded.weight[:old_num_embeddings].copy_(old_weight)
        mean_row = old_weight.mean(dim=0, keepdim=True)
        expanded.weight[old_num_embeddings:].copy_(mean_row.expand(new_num_embeddings - old_num_embeddings, -1))
    return expanded


# ── ORM model wrapper ─────────────────────────────────────────────────────────
class OutcomeRewardModel:
    """Unified wrapper exposing ``score_pairs`` for every supported ORM backend."""

    def __init__(self, spec: dict[str, object], load_in_4bit: bool, torch_dtype=torch.bfloat16):
        self.label = str(spec["label"])
        self.model_name = str(spec["model_name"])
        self.backend = str(spec["backend"])
        quant_config = _make_quant_config(load_in_4bit)
        common = dict(
            quantization_config=quant_config,
            torch_dtype=torch_dtype if not load_in_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.backend == "seqcls":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=1, **common
            )
        elif self.backend in ("internlm2", "eurus"):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            config = _load_sanitized_config(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name, config=config, **common)
        elif self.backend == "rlhflow_orm":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **common)
        else:
            raise ValueError(f"Unsupported ORM backend: {self.backend}")

        # InternLM2 reward repos register their chat special tokens (<|im_start|>,
        # <|im_end|>, ...) as *added tokens* at ids beyond the checkpoint input
        # embedding. We must expand only the input embedding rows here: calling
        # resize_token_embeddings() also resizes get_output_embeddings(), and for
        # this reward model that hook points at the 1-dim reward head (v_head),
        # which breaks scoring.
        if self.backend == "internlm2":
            embedding_rows = self.model.get_input_embeddings().weight.shape[0]
            if len(self.tokenizer) > embedding_rows:
                expanded_embeddings = _expand_embedding_rows(
                    self.model.get_input_embeddings(),
                    len(self.tokenizer),
                )
                self.model.set_input_embeddings(expanded_embeddings)
                self.model.config.vocab_size = len(self.tokenizer)
                self.model.vocab_size = len(self.tokenizer)
                self.model.model.vocab_size = len(self.tokenizer)

        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.backend == "rlhflow_orm":
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @torch.no_grad()
    def _score_seqcls(self, prompt: str, response: str) -> float:
        conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conv, tokenize=True, return_tensors="pt"
        ).to(self.model.device)
        return float(self.model(input_ids).logits[0][0].item())

    @torch.no_grad()
    def _score_internlm2(self, prompt: str, response: str) -> float:
        return self._score_internlm2_batch([prompt], [response])[0]

    @torch.no_grad()
    def _score_internlm2_batch(self, prompts: list[str], responses: list[str]) -> list[float]:
        conversations = [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            for prompt, response in zip(prompts, responses)
        ]
        batch_input_ids = []
        attention_masks = []

        for conversation in conversations:
            conversation_text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            input_ids = self.tokenizer.encode(
                conversation_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            if input_ids[0, -1] != self.model.reward_token_id:
                reward_token = torch.tensor(
                    [[self.model.reward_token_id]],
                    dtype=torch.long,
                )
                input_ids = torch.cat([input_ids, reward_token], dim=1)
            batch_input_ids.append(input_ids.squeeze(0))
            attention_masks.append(torch.ones(input_ids.shape[1], dtype=torch.bool))

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(self.model.device)
        padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=False,
        ).to(self.model.device)

        outputs = self.model.model(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_masks,
            return_dict=True,
        )
        reward_logits = self.model.v_head(outputs[0]).squeeze(-1)
        last_token_indices = padded_attention_masks.long().sum(dim=1) - 1
        scores = reward_logits.gather(1, last_token_indices.unsqueeze(1)).squeeze(1)
        return [_coerce_score_scalar(score) for score in scores.float().detach().cpu().tolist()]

    @torch.no_grad()
    def _score_eurus(self, prompt: str, response: str) -> float:
        # Eurus uses the Mistral-Instruct-v0.2 template (per the official card).
        text = f"[INST] {prompt.strip()} [/INST] {response.strip()}"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        return float(self.model(**inputs).item())

    @torch.no_grad()
    def _score_rlhflow_orm(self, prompt: str, response: str) -> float:
        # RLHFlow math ORM evaluation scores the probability of '+' on the assistant
        # sign turn after concatenating question and full reasoning trace into one user turn.
        conversation = [
            {"role": "user", "content": f"{prompt.strip()} {response.strip()}".strip()},
            {"role": "assistant", "content": "+"},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt",
            return_dict=False,
        ).to(self.model.device)
        plus_tag_id = self.tokenizer.encode("+", add_special_tokens=False)[-1]
        minus_tag_id = self.tokenizer.encode("-", add_special_tokens=False)[-1]
        candidate_tokens = [plus_tag_id, minus_tag_id]
        logits = self.model(input_ids).logits[:, -3, candidate_tokens]
        score = logits.softmax(dim=-1)[:, 0]
        return float(score[0].detach().to("cpu", dtype=torch.float32).item())

    def score_pairs(self, prompts: list[str], responses: list[str]) -> list[float]:
        if self.backend == "internlm2":
            return self._score_internlm2_batch(prompts, responses)
        scorer = {
            "seqcls": self._score_seqcls,
            "internlm2": self._score_internlm2,
            "rlhflow_orm": self._score_rlhflow_orm,
            "eurus": self._score_eurus,
        }[self.backend]
        return [scorer(prompt, response) for prompt, response in zip(prompts, responses)]


def _pick_best_of_n(responses: list[str], scores: list[float]) -> tuple[int, str, float]:
    if len(responses) != len(scores):
        raise ValueError(f"Mismatched counts: {len(responses)} responses vs {len(scores)} scores")
    if not responses:
        raise ValueError("Cannot select best-of-n from an empty candidate list")
    best_index = max(range(len(responses)), key=lambda idx: scores[idx])
    return best_index, responses[best_index], scores[best_index]


def _coerce_score_scalar(value: object) -> float:
    current = value
    while isinstance(current, list):
        if len(current) != 1:
            raise ValueError(f"Expected scalar ORM score, got nested list with {len(current)} entries")
        current = current[0]
    return float(current)


# ── policy sampling (mirrors test_prm_samples.py) ─────────────────────────────
def _load_policy_model(model_name: str, force_4bit: bool):
    tokenizer = _load_policy_tokenizer(model_name)
    load_4bit = force_4bit or _needs_4bit(model_name)
    quant_config = _make_quant_config(load_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if not load_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def sample_responses(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    responses: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
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
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for seq in generated:
            response_ids = _trim_generated_ids(
                seq[input_len:].tolist(),
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
            )
            responses.append(tokenizer.decode(response_ids, skip_special_tokens=True).strip())
    return responses


# ── printing ──────────────────────────────────────────────────────────────────
def _print_example_header(example_idx: int, total_examples: int, policy_model_name: str) -> None:
    print("=" * 100)
    print(f"Example {example_idx}/{total_examples}")
    print("-" * 100)
    print(f"Policy model: {policy_model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small ORM smoke test on local math datasets.")
    parser.add_argument("--dataset", choices=sorted(DEFAULT_DATASETS), default="gsm8k")
    parser.add_argument("--data-csv", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--policy-model", dest="policy_models", action="append")
    parser.add_argument("--policy-preset", choices=sorted(POLICY_PRESETS), default=None)
    parser.add_argument("--orm-model", dest="orm_model_names", action="append")
    parser.add_argument("--orm-preset", dest="orm_presets", action="append", choices=sorted(ORM_PRESETS))
    parser.add_argument(
        "--orm-default-backend",
        choices=["seqcls", "internlm2", "eurus"],
        default="seqcls",
        help="Decode backend to use for any custom --orm-model passed by name.",
    )
    parser.add_argument("--max-rows", type=int, default=5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--policy-load-in-4bit", action="store_true")
    parser.add_argument("--orm-load-in-4bit", action="store_true")
    parser.add_argument("--score-inline-solutions", action="store_true")
    parser.add_argument("--use-inline-solutions", action="store_true")
    parser.add_argument("--score-curated-traces", action="store_true")
    parser.add_argument("--curated-dataset", choices=["gsm8k", "eedi", "both"], default="both")
    parser.add_argument("--curated-case-id", default=None)
    parser.add_argument("--manual-two-traces", action="store_true")
    parser.add_argument("--manual-correct-trace", default=None)
    parser.add_argument("--manual-wrong-trace", default=None)
    parser.add_argument("--list-options", action="store_true")
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> pd.DataFrame:
    if _use_curated_trace_mode(args):
        curated_cases = _filter_curated_cases_by_id(
            _get_curated_trace_cases(args.curated_dataset),
            args.curated_case_id,
        )
        return pd.DataFrame(
            [
                {
                    "dataset": case["dataset"],
                    "case_id": case["case_id"],
                    "question": case["question"],
                    "prompt": case["question"],
                    "split": "curated",
                }
                for case in curated_cases
            ]
        )
    if _use_inline_solution_mode(args):
        return pd.DataFrame([{"question": INLINE_PROBLEM, "prompt": INLINE_PROBLEM, "split": "inline"}])
    data_csv = Path(args.data_csv) if args.data_csv else DEFAULT_DATASETS[args.dataset]
    df = pd.read_csv(data_csv)
    if "split" in df.columns:
        df = df[df["split"] == args.split].copy()
    return df.head(args.max_rows).reset_index(drop=True)


def _validate_args(args: argparse.Namespace) -> None:
    if _use_curated_trace_mode(args) or _use_inline_solution_mode(args):
        return
    if args.manual_two_traces:
        if not args.manual_correct_trace or not args.manual_wrong_trace:
            raise ValueError(
                "--manual-two-traces requires both --manual-correct-trace and --manual-wrong-trace"
            )
        if args.max_rows != 1:
            raise ValueError("--manual-two-traces expects exactly one dataset problem, so use --max-rows 1")


def main() -> None:
    args = parse_args()
    if args.list_options:
        _print_available_options()
        return
    _validate_args(args)

    df = load_rows(args)
    prompts = [_build_prompt(row, "correct_answer") for _, row in df.iterrows()]

    policy_models = _expand_policy_models(
        [name for name in [args.model_name, *(args.policy_models or [])] if name],
        args.policy_preset,
    )
    orm_specs = _resolve_orm_specs(
        args.orm_model_names,
        args.orm_presets,
        args.orm_default_backend,
    )

    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    total_examples = len(df)
    if _use_inline_solution_mode(args) and total_examples != 1:
        raise ValueError("Inline solution mode expects exactly one inline problem")

    max_new_tokens = args.max_new_tokens or TASK_MAX_NEW_TOKENS["correct_answer"]

    run_policy_models = [policy_models[0]] if _use_inline_solution_mode(args) else policy_models
    if _use_curated_trace_mode(args):
        run_policy_models = ["curated_manual_traces"]

    curated_cases = (
        _filter_curated_cases_by_id(
            _get_curated_trace_cases(args.curated_dataset),
            args.curated_case_id,
        )
        if _use_curated_trace_mode(args)
        else []
    )

    for policy_model_name in run_policy_models:
        print("#" * 100)
        if _use_inline_solution_mode(args):
            print("Scoring inline solutions with ORM")
        elif _use_curated_trace_mode(args):
            print("Scoring curated correct/wrong traces with ORM")
        else:
            print(f"Sampling with policy model: {policy_model_name}")
        print("#" * 100)
        print()

        if _use_curated_trace_mode(args):
            repeated_prompts: list[str] = []
            sampled_responses: list[str] = []
            for case in curated_cases:
                repeated_prompts.extend([case["question"], case["question"]])
                sampled_responses.extend(
                    [
                        _normalize_curated_trace_text(case["correct_trace"]),
                        _normalize_curated_trace_text(case["wrong_trace"]),
                    ]
                )
            candidate_count = 2
            grouped_responses = [sampled_responses[i * 2 : (i + 1) * 2] for i in range(total_examples)]
        elif _use_inline_solution_mode(args):
            repeated_prompts = [prompts[0] for _ in INLINE_SOLUTIONS]
            sampled_responses = [solution.strip() for solution in INLINE_SOLUTIONS]
            candidate_count = len(INLINE_SOLUTIONS)
            grouped_responses = [sampled_responses]
        elif args.manual_two_traces:
            repeated_prompts = [prompts[0], prompts[0]]
            sampled_responses = [args.manual_correct_trace.strip(), args.manual_wrong_trace.strip()]
            candidate_count = 2
            grouped_responses = [sampled_responses]
        else:
            model, tokenizer = _load_policy_model(policy_model_name, args.policy_load_in_4bit)
            repeated_prompts = [prompt for prompt in prompts for _ in range(args.num_samples)]
            sampled_responses = sample_responses(
                model,
                tokenizer,
                repeated_prompts,
                batch_size=args.batch_size,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=max_new_tokens,
            )
            del model
            torch.cuda.empty_cache()
            candidate_count = args.num_samples
            grouped_responses = [
                sampled_responses[i * args.num_samples : (i + 1) * args.num_samples]
                for i in range(total_examples)
            ]

        scores_by_name: dict[str, list[float]] = {}
        best_selection_by_name: dict[str, list[tuple[int, str, float]]] = {}
        for spec in orm_specs:
            print(f"Scoring with ORM: {spec['label']} ({spec['model_name']}) backend={spec['backend']}")
            orm = OutcomeRewardModel(spec, load_in_4bit=args.orm_load_in_4bit)
            scores = [_coerce_score_scalar(score) for score in orm.score_pairs(repeated_prompts, sampled_responses)]
            scores_by_name[str(spec["model_name"])] = scores

            grouped_scores = [
                scores[i * candidate_count : (i + 1) * candidate_count]
                for i in range(len(grouped_responses))
            ]
            best_selection_by_name[str(spec["model_name"])] = [
                _pick_best_of_n(candidate_responses, candidate_scores)
                for candidate_responses, candidate_scores in zip(grouped_responses, grouped_scores)
            ]
            del orm
            torch.cuda.empty_cache()

        for idx, (row, prompt, candidate_responses) in enumerate(
            zip(df.itertuples(index=False), prompts, grouped_responses), start=1
        ):
            _print_example_header(idx, total_examples, policy_model_name)
            print("Question:")
            print(getattr(row, "question", getattr(row, "prompt", prompt)))
            print()
            if _use_curated_trace_mode(args):
                print(f"Dataset: {getattr(row, 'dataset', '')}")
                print(f"Case ID: {getattr(row, 'case_id', '')}")
                print()
            for spec in orm_specs:
                spec_key = str(spec["model_name"])
                candidate_scores = scores_by_name[spec_key][
                    (idx - 1) * candidate_count : idx * candidate_count
                ]
                best_index, best_response, best_score = best_selection_by_name[spec_key][idx - 1]
                print(f"ORM: {spec['label']} ({spec['backend']})")
                print(f"  selected candidate: {best_index + 1}/{candidate_count} (score={best_score:.6f})")
                print()
                for candidate_idx, (response, score) in enumerate(
                    zip(candidate_responses, candidate_scores), start=1
                ):
                    if _use_curated_trace_mode(args) or args.manual_two_traces:
                        candidate_label = "Correct trace" if candidate_idx == 1 else "Wrong trace"
                    elif _use_inline_solution_mode(args):
                        candidate_label = f"Inline solution {candidate_idx}/{candidate_count}"
                    else:
                        candidate_label = f"Candidate {candidate_idx}/{candidate_count}"
                    steps = _split_reasoning_steps(response)
                    marker = " <== selected" if candidate_idx - 1 == best_index else ""
                    print(f"{candidate_label}: ORM score = {score:.6f}{marker}")
                    print("Response:")
                    print(response)
                    print("Separated steps:")
                    for step_idx, step in enumerate(steps, start=1):
                        print(f"  Step {step_idx}: {step}")
                    print()
                print("Best-of-N selected solution:")
                print(best_response)
                print()


if __name__ == "__main__":
    main()
