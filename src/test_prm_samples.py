#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from prm_post_train import (
    DEFAULT_PRM_MODEL,
    DEFAULT_PRM_SYSTEM_PROMPT,
    QwenMathProcessRewardModel,
    STEP_SEPARATOR,
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
PRM_PRESETS = {
    "qwen_math_7b": {
        "label": "Qwen Math PRM 7B",
        "backend": "qwen_math",
        "model_name": "Qwen/Qwen2.5-Math-PRM-7B",
        "system_prompt": DEFAULT_PRM_SYSTEM_PROMPT,
        "supported": True,
        "default_reward_field": "reward_mean_log",
        "notes": "Native fit for the current Qwen-style step-classifier path.",
    },
    "skywork_prm_1p5b": {
        "label": "Skywork o1 Open PRM 1.5B",
        "backend": "custom",
        "model_name": "Skywork/Skywork-o1-Open-PRM-Qwen2.5-1.5B",
        "system_prompt": DEFAULT_PRM_SYSTEM_PROMPT,
        "supported": False,
        "default_reward_field": "reward_mean_log",
        "notes": "Published math PRM, but its inference path differs from the current Qwen scorer.",
    },
    "skywork_prm_7b": {
        "label": "Skywork o1 Open PRM 7B",
        "backend": "custom",
        "model_name": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
        "system_prompt": DEFAULT_PRM_SYSTEM_PROMPT,
        "supported": False,
        "default_reward_field": "reward_mean_log",
        "notes": "Published math PRM, but its inference path differs from the current Qwen scorer.",
    },
    "rlhflow_prm_8b": {
        "label": "RLHFlow DeepSeek PRM 8B",
        "backend": "rlhflow_mathrm",
        "model_name": "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        "system_prompt": DEFAULT_PRM_SYSTEM_PROMPT,
        "supported": True,
        "default_reward_field": "reward_mean",
        "notes": "Official RLHFlow math-rm PRM interface. Scores each step with P(+), then averages over steps.",
    },
    "rlhflow_prm_mistral_8b": {
        "label": "RLHFlow Mistral PRM 8B",
        "backend": "rlhflow_mathrm",
        "model_name": "RLHFlow/Llama3.1-8B-PRM-Mistral-Data",
        "system_prompt": DEFAULT_PRM_SYSTEM_PROMPT,
        "supported": True,
        "default_reward_field": "reward_mean",
        "notes": "Official RLHFlow math-rm PRM interface trained on Mistral-style process data.",
    },
    "openr_math_psa_7b": {
        "label": "OpenR MATH PSA PRM 7B",
        "backend": "custom",
        "model_name": "openreasoner/Math-psa",
        "system_prompt": DEFAULT_PRM_SYSTEM_PROMPT,
        "supported": False,
        "default_reward_field": "reward_mean_log",
        "notes": "Referenced by other math-PRM evaluations, but not wired into this script yet.",
    },
}

META_STEP_PATTERNS = [
    r"^let'?s solve",
    r"^let us solve",
    r"^step[- ]by[- ]step",
    r"^we need to",
    r"^we can",
    r"^to solve",
    r"^to find",
    r"^therefore$",
    r"^therefore,?$",
    r"^the answer is",
]

MANUAL_TRACE_CASES = {
    "janet_eggs": {
        "question": (
            "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning "
            "and bakes muffins for her friends every day with 4. She sells the remainder "
            "at the farmers' market daily for $2 per fresh duck egg. How much in dollars "
            "does she make every day at the farmers' market?"
        ),
        "correct_steps": [
            "Janet lays 16 eggs each day.",
            "She uses 3 eggs for breakfast and 4 eggs for muffins.",
            "She uses 3 + 4 = 7 eggs before selling.",
            "She has 16 - 7 = 9 eggs left to sell.",
            "She earns 9 * 2 = 18 dollars each day.",
        ],
        "wrong_steps": [
            "Janet lays 16 eggs each day.",
            "She uses 3 eggs for breakfast and 4 eggs for muffins.",
            "She uses 3 + 4 = 7 eggs before selling.",
            "She has 16 + 7 = 23 eggs left to sell.",
            "She earns 23 * 2 = 46 dollars each day.",
        ],
    },
}
INLINE_PROBLEM = (
    "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning "
    "and bakes muffins for her friends every day with 4. She sells the remainder "
    "at the farmers' market daily for $2 per fresh duck egg. How much in dollars "
    "does she make every day at the farmers' market?"
)
INLINE_SOLUTIONS = [
    "\n\n".join(
        [
            "She uses 3 + 4 = 7 eggs before selling.",
            "She has 16 - 7 = 9 eggs left to sell.",
            "She earns 9 * 2 = 18 dollars each day.",
        ]
    ),
    "\n\n".join(
        [
            "Janet lays 16 eggs each day.",
            "She uses 3 eggs for breakfast and 4 eggs for muffins.",
            "She uses 3 + 4 = 7 eggs before selling.",
            "She has 16 + 7 = 23 eggs left to sell.",
            "She earns 23 * 2 = 46 dollars each day.",
        ]
    ),
    "\n\n".join(
        [
            "Janet lays 16 eggs each day.",
            "Thus she earns -16 dollars each day.",
        ]
    ),
]

CURATED_TRACE_CASES = [
    {
        "dataset": "gsm8k",
        "case_id": "gsm8k_1_natalia_clips",
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "correct_trace": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.",
        "wrong_trace": "Natalia sold half as many clips in May, so she sold 48 * 2 = 96 clips in May.\nAltogether she sold 48 + 96 = 144 clips.",
    },
    {
        "dataset": "gsm8k",
        "case_id": "gsm8k_2_weng_babysitting",
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "correct_trace": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.",
        "wrong_trace": "Weng earns $12 in 60 minutes.\nIn 50 minutes, she earns 12 * 50 = $600.",
    },
    {
        "dataset": "gsm8k",
        "case_id": "gsm8k_3_betty_wallet",
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "correct_trace": "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.",
        "wrong_trace": "Betty starts with $50.\nHer parents and grandparents give her 15 + 30 = $45 more, so now she has $95.\nShe still needs 100 - 95 = $15.",
    },
    {
        "dataset": "gsm8k",
        "case_id": "gsm8k_4_julie_book",
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "correct_trace": "Yesterday, Julie read 12 pages and today, she read twice as many pages as yesterday, thus Julie read 12 x 2 = 24 pages today.\nSo she was able to read a total of 12 + 24 = 36 pages since yesterday.\nThere are 120 - 36 = <<120-36=84>>84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.",
        "wrong_trace": "Yesterday, Julie read 12 pages and today, she read twice as many pages as yesterday, thus Julie read 12 x 2 = 24 pages today.\nThat means she has read 24 pages total.\nShe has 120 - 24 = 96 pages left.\nSince she wants to read half of the remaining pages tomorrow, she will have to read 96/2=48 pages.",
    },
    {
        "dataset": "gsm8k",
        "case_id": "gsm8k_5_james_letters",
        "question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?",
        "correct_trace": "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year",
        "wrong_trace": "James writes 3 pages to 2 friends, so that is 3 + 2 = 5 pages each week.\nTwice a week means 10 pages a week.\nOver a year, he writes 10 * 52 = 520 pages.",
    },
    {
        "dataset": "eedi",
        "case_id": "eedi_1_brackets_13",
        "question": "\\[\n3 \\times 2+4-5\n\\]\nWhere do the brackets need to go to make the answer equal \\( 13 \\) ?",
        "correct_trace": "Use order of operations and try grouping the addition so it happens before multiplication.\n\nWithout brackets:\n\\[\n3\\times 2+4-5=6+4-5=5\n\\]\nThat is not \\(13\\).\n\nIf we put brackets around \\(2+4\\), then:\n\\[\n3\\times(2+4)-5=3\\times 6-5=18-5=13\n\\]\n\nSo the brackets should go as:\n\\[\n3 \\times(2+4)-5\n\\]",
        "wrong_trace": "To make the answer 13, put brackets around 4-5 first.\nThen 3 times 2 plus (4-5) becomes 6 + (-1) = 13.\nSo the correct choice is 3 times 2 plus (4-5).",
    },
    {
        "dataset": "eedi",
        "case_id": "eedi_2_does_not_simplify",
        "question": "Simplify the following, if possible: \\( \\frac{m^{2}+2 m-3}{m-3} \\)",
        "correct_trace": "To simplify a rational expression, we look for common factors in the numerator and denominator.\n\nFirst factor the numerator:\n\\[\nm^2+2m-3\n\\]\nWe need two numbers that multiply to \\(-3\\) and add to \\(2\\). Those numbers are \\(3\\) and \\(-1\\), so:\n\\[\nm^2+2m-3=(m+3)(m-1)\n\\]\n\nSo the expression becomes:\n\\[\n\\frac{(m+3)(m-1)}{m-3}\n\\]\n\nThe denominator is \\(m-3\\), and there is no matching factor of \\(m-3\\) in the numerator, so no cancellation is possible.\n\nTherefore, the expression does not simplify. The correct answer is: Does not simplify",
        "wrong_trace": "Factor the numerator as (m+3)(m-1).\nThen cancel the m-3 in the denominator with the m-1 in the numerator.\nSo the expression simplifies to m+3.",
    },
    {
        "dataset": "eedi",
        "case_id": "eedi_3_table_star_30",
        "question": "The equation \\( f=3 r^{2}+3 \\) is used to find values in the table below. What is the value covered by the star? \\begin{tabular}{|c|c|c|c|c|}\n\\hline\\( r \\) & \\( 1 \\) & \\( 2 \\) & \\( 3 \\) & \\( 4 \\) \\\\n\\hline\\( f \\) & \\( 6 \\) & \\( 15 \\) & \\( \\color{gold}\\bigstar \\) & \\\\n\\hline\n\\end{tabular}",
        "correct_trace": "Use the equation \\(f = 3r^2 + 3\\). The star is in the row where \\(r = 3\\), so substitute \\(3\\) for \\(r\\):\n\n\\[\nf = 3(3^2) + 3\n\\]\n\nFirst, compute \\(3^2 = 9\\):\n\n\\[\nf = 3(9) + 3 = 27 + 3 = 30\n\\]\n\nThe correct answer is \\(30\\).",
        "wrong_trace": "Since r = 3, substitute directly into 3r^2 + 3 as 3 + 3^2 = 6^2 = 36.\nSo the value under the star is 36.",
    },
    {
        "dataset": "eedi",
        "case_id": "eedi_4_area_units",
        "question": "James has answered a question on the area of a trapezium and got an answer of \\( 54 \\).\n\nBehind the star he has written the units that he used.\n\n\\(\n54 \\, \\bigstar \n\\)\n\nWhich of the following units could be correct?",
        "correct_trace": "Area is a measure of surface, so its units must be **square units**.\n\nThat means valid area units look like:\n- \\(\\text{mm}^2\\)\n- \\(\\text{cm}^2\\)\n- \\(\\text{m}^2\\)\n\nUnits that are just length, like \\(\\text{mm}\\) or \\(\\text{cm}\\), would not be correct for area.\n\nSo a unit that **could** be correct for \\(54\\) as an area is \\(\\mathrm{mm}^2\\).\n\nThe correct answer is \\(\\mathrm{mm}^{2}\\).",
        "wrong_trace": "Because a trapezium has edges, area should be measured in millimetres.\nSo the unit could be mm, not mm squared.",
    },
    {
        "dataset": "eedi",
        "case_id": "eedi_5_percent_fraction",
        "question": "Convert this percentage to a fraction\n\\( 62 \\% \\)",
        "correct_trace": "A percent means “out of 100,” so \\(62\\%\\) = \\(\\frac{62}{100}\\).\n\nNow simplify the fraction by dividing the numerator and denominator by 2:\n\n\\[\n\\frac{62}{100} = \\frac{31}{50}\n\\]\n\nSo the correct answer is \\( \\frac{31}{50} \\).",
        "wrong_trace": "62% means 62 out of 10, so the fraction is 62/10.\nSimplifying gives 31/5.",
    },
]

INLINE_PROBLEM = (
    "Solve for x: 2^(x+1) + 2^x = 24."
)

INLINE_SOLUTIONS = [
    # Correct solution
    "\n\n".join(
        [
            "We use the identity 2^(x+1) = 2 * 2^x.",
            "Therefore, 2^(x+1) + 2^x = 2 * 2^x + 2^x = 3 * 2^x.",
            "So the equation becomes 3 * 2^x = 24.",
            "Dividing both sides by 3 gives 2^x = 8.",
            "Since 8 = 2^3, we get x = 3.",
        ]
    ),

    # Wrong solution: invalid exponent manipulation
    "\n\n".join(
        [
            "We combine the two powers by adding their exponents.",
            "Thus 2^(x+1) + 2^x = 2^((x+1)+x) = 2^(2x+1).",
            "So the equation becomes 2^(2x+1) = 24.",
            "Taking log base 2 gives 2x + 1 = log_2(24).",
            "Therefore x = (log_2(24) - 1) / 2.",
        ]
    ),

    # Very wrong solution: arithmetic / algebra nonsense
    "\n\n".join(
        [
            "We subtract the exponents and get 2^(x+1) + 2^x = 2.",
            "So the equation becomes 2 = 24.",
            "Since this is false, there is no solution.",
        ]
    ),
]


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
    filtered_cases = [case for case in curated_cases if case["case_id"] == case_id]
    if not filtered_cases:
        raise ValueError(f"No curated trace case found for --curated-case-id {case_id!r}")
    return filtered_cases


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


def _resolve_prm_specs(
    prm_model_names: list[str] | None,
    prm_presets: list[str] | None,
    default_system_prompt: str,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for preset_name in prm_presets or []:
        preset = PRM_PRESETS[preset_name]
        specs.append(
            {
                "label": preset["label"],
                "backend": preset["backend"],
                "model_name": preset["model_name"],
                "system_prompt": preset["system_prompt"],
                "supported": preset["supported"],
                "default_reward_field": preset["default_reward_field"],
                "notes": preset["notes"],
            }
        )
    for model_name in prm_model_names or []:
        specs.append(
            {
                "label": model_name,
                "backend": "qwen_math",
                "model_name": model_name,
                "system_prompt": default_system_prompt,
                "supported": True,
                "default_reward_field": "reward_mean_log",
                "notes": "Custom PRM model name passed directly on the CLI.",
            }
        )
    if not specs:
        default_preset = PRM_PRESETS["qwen_math_7b"]
        specs.append(
            {
                "label": default_preset["label"],
                "backend": default_preset["backend"],
                "model_name": default_preset["model_name"],
                "system_prompt": default_preset["system_prompt"],
                "supported": default_preset["supported"],
                "default_reward_field": default_preset["default_reward_field"],
                "notes": default_preset["notes"],
            }
        )

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
    print("PRM presets:")
    for name, spec in PRM_PRESETS.items():
        support = "supported" if spec["supported"] else "catalogued_only"
        print(
            f"  {name}: backend={spec['backend']} status={support} model={spec['model_name']}"
        )
        print(f"    {spec['notes']}")


def _normalize_steps(response: str, filter_meta_steps: bool) -> list[str]:
    raw_steps = _split_reasoning_steps(response)
    if not filter_meta_steps:
        return raw_steps
    filtered = []
    for step in raw_steps:
        compact = re.sub(r"\s+", " ", step.strip()).lower()
        if any(re.match(pattern, compact) for pattern in META_STEP_PATTERNS):
            continue
        filtered.append(step)
    return filtered or raw_steps


def _build_prm_conversation_text(
    prm: QwenMathProcessRewardModel,
    prompt: str,
    response: str,
    step_format: str,
    filter_meta_steps: bool,
) -> tuple[str, list[str]]:
    steps = _normalize_steps(response, filter_meta_steps)
    if step_format == "compact":
        assistant_content = STEP_SEPARATOR.join(steps) + STEP_SEPARATOR
    else:
        assistant_content = "\n\n".join(f"{step.strip()} {STEP_SEPARATOR}" for step in steps)
    messages = [
        {"role": "system", "content": prm.system_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_content},
    ]
    return (
        prm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ),
        steps,
    )


def _split_steps_for_backend(response: str, backend: str) -> list[str]:
    if backend == "rlhflow_mathrm":
        return [step.strip() for step in response.split("\n\n") if step.strip()]
    return _split_reasoning_steps(response)


def _build_rlhflow_thinkprm_conversations(
    prompt: str,
    response: str,
    filter_meta_steps: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Build the ThinkPRM-style labeled conversation plus a parallel masking conversation.

    Each reasoning step becomes a user turn, followed by an assistant turn holding the
    label token. The real conversation uses "+" as the assistant label, while the mask
    conversation uses the dummy token "ки" so the assistant-label positions can be located
    after tokenization.
    """
    steps = _normalize_steps(response, filter_meta_steps)
    conversation: list[dict[str, str]] = []
    conversation_mask: list[dict[str, str]] = []
    for step_idx, step in enumerate(steps):
        if step_idx == 0:
            user_content = f"{prompt.strip()} {step.strip()}".strip()
        else:
            user_content = step.strip()
        conversation.append({"content": user_content, "role": "user"})
        conversation.append({"content": "+", "role": "assistant"})
        conversation_mask.append({"content": user_content, "role": "user"})
        conversation_mask.append({"content": "ки", "role": "assistant"})
    return conversation, conversation_mask, steps


def _aggregate_step_scores_for_backend(
    step_scores: list[float],
    positive_class_index: int,
    backend: str,
) -> dict[str, float | int | str]:
    aggregates = _aggregate_step_scores(step_scores, positive_class_index)
    aggregates["backend"] = backend
    if backend == "rlhflow_mathrm":
        aggregates["full_prefix_score"] = aggregates["reward_mean"]
    else:
        aggregates["full_prefix_score"] = step_scores[-1] if step_scores else float("nan")
    return aggregates


class RLHFFlowMathPRM:
    def __init__(self, model_name: str, load_in_4bit: bool, torch_dtype=torch.bfloat16):
        self.model_name = model_name
        self.backend = "rlhflow_mathrm"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        quant_config = _make_quant_config(load_in_4bit)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=torch_dtype if not load_in_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
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


def _print_rlhflow_debug_snapshot(
    prm: RLHFFlowMathPRM,
    conversation: list[dict[str, str]],
    inputs: torch.Tensor,
    inputs_mask: torch.Tensor,
    label_positions: list[int],
    class_probabilities: list[list[float]],
) -> None:
    labeled_text = prm.tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
    )
    print("RLHFlow ThinkPRM-style labeled conversation used for scoring")
    print(labeled_text)
    print(f"candidate token ids: {prm.candidate_tokens} (index 0 = '+', index 1 = '-')")
    print(f"dummy mask token 'ки' id: {prm.special_tok_id}")
    print(f"inputs shape: {tuple(inputs.shape)} inputs_mask shape: {tuple(inputs_mask.shape)}")
    print(f"assistant-label positions (shifted by -1 for causal LM): {label_positions}")
    for step_idx, (pos, probs) in enumerate(zip(label_positions, class_probabilities), start=1):
        print(f"    Step {step_idx}: score_pos={pos} p_plus={probs[0]:.6f} p_minus={probs[1]:.6f}")
    print()


def _score_with_rlhflow_debug(
    prm: RLHFFlowMathPRM,
    prompts: list[str],
    responses: list[str],
    batch_size: int,
    filter_meta_steps: bool,
    debug_prm: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_results: list[dict[str, object]] = []
    all_debug: list[dict[str, object]] = []

    for prompt, response in zip(prompts, responses):
        conversation, conversation_mask, normalized_steps = _build_rlhflow_thinkprm_conversations(
            prompt, response, filter_meta_steps
        )

        if not conversation:
            aggregates = _aggregate_step_scores_for_backend([], positive_class_index=0, backend=prm.backend)
            all_results.append({**aggregates, "step_scores": []})
            all_debug.append(
                {
                    "conversation": "",
                    "normalized_steps": normalized_steps,
                    "class_probabilities": [],
                    "num_separators": 0,
                    "logits_shape": (0, 2),
                }
            )
            continue

        inputs = prm.tokenizer.apply_chat_template(
            conversation,
            return_tensors="pt",
            return_dict=False,
        ).to(prm.model.device)
        inputs_mask = prm.tokenizer.apply_chat_template(
            conversation_mask,
            return_tensors="pt",
            return_dict=False,
        ).to(prm.model.device)
        if inputs.shape != inputs_mask.shape:
            raise ValueError(
                f"RLHFlow mask conversation shape {tuple(inputs_mask.shape)} does not match "
                f"label conversation shape {tuple(inputs.shape)}"
            )

        with torch.no_grad():
            # logits at position t predict token t+1, so the assistant-label probability
            # lives at the position before the dummy/"+"" token.
            logits = prm.model(inputs).logits[:, :, prm.candidate_tokens]
            probs = logits.softmax(dim=-1)

        label_mask = inputs_mask[0, 1:] == prm.special_tok_id
        selected_probs = probs[0, :-1][label_mask]
        class_probabilities = selected_probs.float().detach().cpu().tolist()
        step_scores = [float(pair[0]) for pair in class_probabilities]
        label_positions = label_mask.nonzero(as_tuple=True)[0].detach().cpu().tolist()

        if debug_prm:
            _print_rlhflow_debug_snapshot(
                prm,
                conversation,
                inputs,
                inputs_mask,
                label_positions,
                class_probabilities,
            )

        serialized_conversation = prm.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        aggregates = _aggregate_step_scores_for_backend(
            step_scores,
            positive_class_index=0,
            backend=prm.backend,
        )
        all_results.append(
            {
                **aggregates,
                "step_scores": step_scores,
            }
        )
        all_debug.append(
            {
                "conversation": serialized_conversation,
                "normalized_steps": normalized_steps,
                "class_probabilities": class_probabilities,
                "num_separators": len(step_scores),
                "logits_shape": (len(step_scores), 2),
            }
        )

    return all_results, all_debug


def _aggregate_step_scores(step_scores: list[float], positive_class_index: int) -> dict[str, float | int]:
    reward_product = 1.0
    reward_logsum = 0.0
    reward_mean = 0.0
    for value in step_scores:
        clipped = max(min(float(value), 1.0), 1e-12)
        reward_product *= clipped
        reward_logsum += math.log(clipped)
        reward_mean += clipped
    num_steps = len(step_scores)
    reward_mean = reward_mean / max(num_steps, 1)
    reward_mean_log = reward_logsum / max(num_steps, 1)
    return {
        "positive_class_index": positive_class_index,
        "reward_product": reward_product,
        "reward_logsum": reward_logsum,
        "reward_mean_log": reward_mean_log,
        "reward_mean": reward_mean,
        "num_steps": num_steps,
    }


def _select_reward_value(prm_result: dict[str, object], reward_field: str) -> float:
    return float(prm_result[reward_field])


def _pick_best_of_n(
    responses: list[str],
    prm_results: list[dict[str, object]],
    reward_field: str,
) -> tuple[int, str, dict[str, object]]:
    if len(responses) != len(prm_results):
        raise ValueError(f"Mismatched candidate counts: {len(responses)} responses vs {len(prm_results)} scores")
    if not responses:
        raise ValueError("Cannot select best-of-n from an empty candidate list")
    best_index = max(range(len(responses)), key=lambda idx: _select_reward_value(prm_results[idx], reward_field))
    return best_index, responses[best_index], prm_results[best_index]


def _resolve_reward_field(aggregate_name: str) -> str:
    mapping = {
        "product": "reward_product",
        "logsum": "reward_logsum",
        "mean_log": "reward_mean_log",
        "mean": "reward_mean",
        "reward_product": "reward_product",
        "reward_logsum": "reward_logsum",
        "reward_mean_log": "reward_mean_log",
        "reward_mean": "reward_mean",
    }
    return mapping[aggregate_name]


def _default_reward_field_for_spec(spec: dict[str, object]) -> str:
    return str(spec.get("default_reward_field", "reward_mean_log"))


def _score_with_debug(
    prm: QwenMathProcessRewardModel,
    prompts: list[str],
    responses: list[str],
    batch_size: int,
    positive_class_index: int,
    step_format: str,
    filter_meta_steps: bool,
    debug_prm: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    conversations_and_steps = [
        _build_prm_conversation_text(prm, prompt, response, step_format, filter_meta_steps)
        for prompt, response in zip(prompts, responses)
    ]
    conversations = [item[0] for item in conversations_and_steps]
    normalized_steps = [item[1] for item in conversations_and_steps]

    all_results: list[dict[str, object]] = []
    all_debug: list[dict[str, object]] = []
    for start in range(0, len(conversations), batch_size):
        batch_texts = conversations[start : start + batch_size]
        batch_steps = normalized_steps[start : start + batch_size]
        enc = prm.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(prm.model.device)
        with torch.no_grad():
            outputs = prm.model(**enc)
        logits = outputs[0]
        token_masks = enc["input_ids"] == prm.step_sep_id
        if debug_prm:
            print("FULL PRM INPUT")
            print(batch_texts[0])
            print(f"input_ids shape: {tuple(enc['input_ids'].shape)}")
            print(f"logits shape: {tuple(logits.shape)}")
            print(f"sep ids: {prm.tokenizer.encode(STEP_SEPARATOR, add_special_tokens=False)}")
            print(f"num {STEP_SEPARATOR}: {int(token_masks[0].sum().item())}")
            print()
        if logits.shape[-1] != 2:
            raise ValueError(f"Expected PRM logits with last dimension 2, got {tuple(logits.shape)}")

        for sample_idx in range(logits.size(0)):
            step_logits = logits[sample_idx, token_masks[sample_idx], :]
            if step_logits.numel() == 0:
                step_probs = torch.empty((0, 2), dtype=torch.float32)
                step_scores: list[float] = []
            else:
                step_probs = F.softmax(step_logits.float(), dim=-1)
                step_scores = step_probs[:, positive_class_index].detach().cpu().tolist()
            aggregates = _aggregate_step_scores(step_scores, positive_class_index)
            class_probs = step_probs.detach().cpu().tolist()
            all_results.append(
                {
                    **aggregates,
                    "step_scores": step_scores,
                }
            )
            all_debug.append(
                {
                    "conversation": batch_texts[sample_idx],
                    "normalized_steps": batch_steps[sample_idx],
                    "class_probabilities": class_probs,
                    "num_separators": int(token_masks[sample_idx].sum().item()),
                    "logits_shape": tuple(logits.shape),
                }
            )
    return all_results, all_debug


def _score_candidates_with_backend(
    prm,
    prompts: list[str],
    responses: list[str],
    batch_size: int,
    positive_class_index: int,
    step_format: str,
    filter_meta_steps: bool,
    debug_prm: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    backend = getattr(prm, "backend", "qwen_math")
    if backend == "qwen_math":
        return _score_with_debug(
            prm,
            prompts,
            responses,
            batch_size=batch_size,
            positive_class_index=positive_class_index,
            step_format=step_format,
            filter_meta_steps=filter_meta_steps,
            debug_prm=debug_prm,
        )
    if backend == "rlhflow_mathrm":
        return _score_with_rlhflow_debug(
            prm,
            prompts,
            responses,
            batch_size=batch_size,
            filter_meta_steps=filter_meta_steps,
            debug_prm=debug_prm,
        )
    raise ValueError(f"Unsupported PRM backend: {backend}")


def _print_prm_debug_block(
    spec: dict[str, object],
    prm_result: dict[str, object],
    prm_debug: dict[str, object],
    compare_class_indices: bool,
) -> None:
    print(f"PRM: {spec['label']}")
    print(f"  model={spec['model_name']}")
    print(f"  backend={spec['backend']}")
    print(f"  positive_class_index={prm_result['positive_class_index']}")
    print(f"  num_separators={prm_debug['num_separators']}")
    print(f"  logits_shape={prm_debug['logits_shape']}")
    print("  step probabilities:")
    for step_idx, probs in enumerate(prm_debug["class_probabilities"], start=1):
        if compare_class_indices:
            print(f"    Step {step_idx}: class0={probs[0]:.6f} class1={probs[1]:.6f}")
        else:
            print(f"    Step {step_idx}: selected={prm_result['step_scores'][step_idx - 1]:.6f}")
    print(
        "  aggregates: "
        f"num_steps={prm_result['num_steps']} "
        f"reward_product={prm_result['reward_product']:.6e} "
        f"reward_logsum={prm_result['reward_logsum']:.6f} "
        f"reward_mean={prm_result['reward_mean']:.6f} "
        f"reward_mean_log={prm_result['reward_mean_log']:.6f}"
    )
    print()


def _run_manual_trace_comparison(
    prm: QwenMathProcessRewardModel,
    args: argparse.Namespace,
    spec: dict[str, object],
) -> None:
    case = MANUAL_TRACE_CASES[args.manual_compare_case]
    prompt = case["question"]
    correct_response = "\n\n".join(case["correct_steps"])
    wrong_response = "\n\n".join(case["wrong_steps"])
    results, debug_rows = _score_candidates_with_backend(
        prm,
        [prompt, prompt],
        [correct_response, wrong_response],
        batch_size=2,
        positive_class_index=args.positive_class_index,
        step_format=args.prm_step_format,
        filter_meta_steps=args.filter_meta_steps,
        debug_prm=args.debug_prm,
    )

    print("=" * 100)
    print(f"Manual PRM comparison: {args.manual_compare_case}")
    print("-" * 100)
    print(prompt)
    print()
    for label, response, result, debug_row in zip(
        ["Correct trace", "Wrong trace"],
        [correct_response, wrong_response],
        results,
        debug_rows,
    ):
        print(label)
        print(response)
        print()
        _print_prm_debug_block(spec, result, debug_row, compare_class_indices=True)


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


def _load_prm_from_spec(spec: dict[str, object], load_in_4bit: bool):
    backend = spec["backend"]
    if backend == "qwen_math":
        scorer = QwenMathProcessRewardModel(
            str(spec["model_name"]),
            system_prompt=str(spec["system_prompt"]),
            load_in_4bit=load_in_4bit,
        )
        scorer.backend = backend
        return scorer
    if backend == "rlhflow_mathrm":
        return RLHFFlowMathPRM(str(spec["model_name"]), load_in_4bit=load_in_4bit)
    raise ValueError(
        f"PRM preset '{spec['label']}' uses backend '{backend}', which this smoke test does not support yet."
    )


def _print_example_header(example_idx: int, total_examples: int, policy_model_name: str) -> None:
    print("=" * 100)
    print(f"Example {example_idx}/{total_examples}")
    print("-" * 100)
    print(f"Policy model: {policy_model_name}")


def _print_prm_block(spec: dict[str, object], prm_result: dict[str, object]) -> None:
    print(f"PRM: {spec['label']}")
    print(f"  model={spec['model_name']}")
    print(f"  backend={spec['backend']}")
    print(f"  positive_class_index={prm_result['positive_class_index']}")
    print("  step scores:")
    for step_idx, score in enumerate(prm_result["step_scores"], start=1):
        print(f"    Step {step_idx}: {score:.6f}")
    print(
        "  aggregates: "
        f"num_steps={prm_result['num_steps']} "
        f"reward_product={prm_result['reward_product']:.6e} "
        f"reward_logsum={prm_result['reward_logsum']:.6f} "
        f"reward_mean={prm_result['reward_mean']:.6f} "
        f"reward_mean_log={prm_result['reward_mean_log']:.6f}"
    )
    print()


def _validate_manual_trace_args(args: argparse.Namespace) -> None:
    if _use_curated_trace_mode(args):
        return
    if _use_inline_solution_mode(args):
        if len(INLINE_SOLUTIONS) < 1:
            raise ValueError("INLINE_SOLUTIONS must contain at least one solution")
        return
    if not args.manual_two_traces:
        return
    if not args.manual_correct_trace or not args.manual_wrong_trace:
        raise ValueError("--manual-two-traces requires both --manual-correct-trace and --manual-wrong-trace")
    if args.max_rows != 1:
        raise ValueError("--manual-two-traces expects exactly one dataset problem, so use --max-rows 1")
    if args.model_name or args.policy_models or args.policy_preset:
        selected_models = [name for name in [args.model_name, *(args.policy_models or [])] if name]
        if args.policy_preset or len(selected_models) > 1:
            raise ValueError("--manual-two-traces expects exactly one policy model selection path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small PRM smoke test on local math datasets.")
    parser.add_argument("--dataset", choices=sorted(DEFAULT_DATASETS), default="gsm8k")
    parser.add_argument("--data-csv", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--policy-model", dest="policy_models", action="append")
    parser.add_argument("--policy-preset", choices=sorted(POLICY_PRESETS), default=None)
    parser.add_argument("--prm-model-name", default=None)
    parser.add_argument("--prm-model", dest="prm_model_names", action="append")
    parser.add_argument("--prm-preset", dest="prm_presets", action="append", choices=sorted(PRM_PRESETS))
    parser.add_argument("--prm-system-prompt", default=DEFAULT_PRM_SYSTEM_PROMPT)
    parser.add_argument("--max-rows", type=int, default=5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--policy-load-in-4bit", action="store_true")
    parser.add_argument("--prm-load-in-4bit", action="store_true")
    parser.add_argument("--positive-class-index", type=int, choices=[0, 1], default=1)
    parser.add_argument("--compare-class-indices", action="store_true")
    parser.add_argument("--debug-prm", action="store_true")
    parser.add_argument("--prm-step-format", choices=["compact", "spaced"], default="spaced")
    parser.add_argument("--filter-meta-steps", action="store_true")
    parser.add_argument("--manual-compare", action="store_true")
    parser.add_argument("--manual-compare-case", choices=sorted(MANUAL_TRACE_CASES), default="janet_eggs")
    parser.add_argument("--manual-two-traces", action="store_true")
    parser.add_argument("--score-inline-solutions", action="store_true")
    parser.add_argument("--use-inline-solutions", action="store_true")
    parser.add_argument("--score-curated-traces", action="store_true")
    parser.add_argument("--curated-dataset", choices=["gsm8k", "eedi", "both"], default="both")
    parser.add_argument("--curated-case-id", default=None)
    parser.add_argument("--manual-correct-trace", default=None)
    parser.add_argument("--manual-wrong-trace", default=None)
    parser.add_argument(
        "--selection-reward-field",
        choices=["reward_product", "reward_logsum", "reward_mean", "reward_mean_log"],
        default=None,
    )
    parser.add_argument(
        "--aggregate",
        choices=["product", "logsum", "mean_log", "mean"],
        default=None,
        help="Alias for --selection-reward-field using the minimal best-of-n naming.",
    )
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


def sample_responses(model, tokenizer, prompts: list[str], batch_size: int, temperature: float, top_p: float, max_new_tokens: int):
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


def main() -> None:
    args = parse_args()
    if args.list_options:
        _print_available_options()
        return
    _validate_manual_trace_args(args)

    df = load_rows(args)
    prompts = [_build_prompt(row, "correct_answer") for _, row in df.iterrows()]

    policy_models = _expand_policy_models(
        [name for name in [args.model_name, *(args.policy_models or [])] if name],
        args.policy_preset,
    )
    prm_model_names = [name for name in [args.prm_model_name, *(args.prm_model_names or [])] if name]
    prm_specs = _resolve_prm_specs(prm_model_names, args.prm_presets, args.prm_system_prompt)

    unsupported_specs = [spec for spec in prm_specs if not spec["supported"]]
    if unsupported_specs:
        print("Skipping catalogued PRMs that are not wired into this smoke test yet:")
        for spec in unsupported_specs:
            print(f"  - {spec['label']} ({spec['model_name']}): {spec['notes']}")
        print()
    runnable_prm_specs = [spec for spec in prm_specs if spec["supported"]]
    if not runnable_prm_specs:
        raise ValueError("No runnable PRM specs selected. Use a supported preset or pass a Qwen-style --prm-model.")
    if args.aggregate:
        reward_fields_by_name = {
            str(spec["model_name"]): _resolve_reward_field(args.aggregate)
            for spec in runnable_prm_specs
        }
    else:
        reward_fields_by_name = {
            str(spec["model_name"]): (
                args.selection_reward_field
                if args.selection_reward_field is not None
                else _default_reward_field_for_spec(spec)
            )
            for spec in runnable_prm_specs
        }
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
            print("Scoring inline solutions with PRM")
        else:
            print(f"Sampling with policy model: {policy_model_name}")
        print("#" * 100)
        print()

        if _use_curated_trace_mode(args):
            repeated_prompts = []
            sampled_responses = []
            for case in curated_cases:
                repeated_prompts.extend([case["question"], case["question"]])
                sampled_responses.extend(
                    [
                        _normalize_curated_trace_text(case["correct_trace"]),
                        _normalize_curated_trace_text(case["wrong_trace"]),
                    ]
                )
        elif _use_inline_solution_mode(args):
            repeated_prompts = [prompts[0] for _ in INLINE_SOLUTIONS]
            sampled_responses = [solution.strip() for solution in INLINE_SOLUTIONS]
        elif args.manual_two_traces:
            if total_examples != 1:
                raise ValueError("--manual-two-traces currently supports exactly one dataset problem")
            repeated_prompts = [prompts[0], prompts[0]]
            sampled_responses = [args.manual_correct_trace.strip(), args.manual_wrong_trace.strip()]
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

        grouped_responses = [
            sampled_responses[i * args.num_samples : (i + 1) * args.num_samples]
            for i in range(total_examples)
        ]
        if _use_curated_trace_mode(args):
            grouped_responses = [sampled_responses[i * 2 : (i + 1) * 2] for i in range(total_examples)]
        elif _use_inline_solution_mode(args):
            grouped_responses = [sampled_responses]
        elif args.manual_two_traces:
            grouped_responses = [sampled_responses]
        candidate_count = 2 if _use_curated_trace_mode(args) else (len(INLINE_SOLUTIONS) if _use_inline_solution_mode(args) else (2 if args.manual_two_traces else args.num_samples))

        prm_results_by_name: dict[str, list[dict[str, object]]] = {}
        prm_debug_by_name: dict[str, list[dict[str, object]]] = {}
        best_selection_by_name: dict[str, list[tuple[int, str, dict[str, object]]]] = {}
        for spec in runnable_prm_specs:
            print(f"Scoring with PRM: {spec['label']} ({spec['model_name']})")
            prm = _load_prm_from_spec(spec, args.prm_load_in_4bit)
            if args.manual_compare:
                _run_manual_trace_comparison(prm, args, spec)
            prm_results, prm_debug = _score_candidates_with_backend(
                prm,
                repeated_prompts,
                sampled_responses,
                batch_size=args.batch_size,
                positive_class_index=args.positive_class_index,
                step_format=args.prm_step_format,
                filter_meta_steps=args.filter_meta_steps,
                debug_prm=args.debug_prm,
            )
            prm_results_by_name[str(spec["model_name"])] = prm_results
            prm_debug_by_name[str(spec["model_name"])] = prm_debug
            grouped_prm_results = [
                prm_results[i * args.num_samples : (i + 1) * args.num_samples]
                for i in range(total_examples)
            ]
            if _use_curated_trace_mode(args):
                grouped_prm_results = [prm_results[i * 2 : (i + 1) * 2] for i in range(total_examples)]
            elif _use_inline_solution_mode(args):
                grouped_prm_results = [prm_results]
            elif args.manual_two_traces:
                grouped_prm_results = [prm_results]
            best_selection_by_name[str(spec["model_name"])] = [
                _pick_best_of_n(
                    candidate_responses,
                    candidate_scores,
                    reward_fields_by_name[str(spec["model_name"])]
                )
                for candidate_responses, candidate_scores in zip(grouped_responses, grouped_prm_results)
            ]
            del prm
            torch.cuda.empty_cache()

        for idx, (row, prompt, candidate_responses) in enumerate(zip(df.itertuples(index=False), prompts, grouped_responses), start=1):
            _print_example_header(idx, total_examples, policy_model_name)
            print("Question:")
            print(getattr(row, "question", getattr(row, "prompt", prompt)))
            print()
            if _use_curated_trace_mode(args):
                print(f"Dataset: {getattr(row, 'dataset', '')}")
                print(f"Case ID: {getattr(row, 'case_id', '')}")
                print()
            for spec in runnable_prm_specs:
                spec_key = str(spec["model_name"])
                grouped_prm_results = prm_results_by_name[spec_key][(idx - 1) * args.num_samples : idx * args.num_samples]
                grouped_prm_debug = prm_debug_by_name[spec_key][(idx - 1) * args.num_samples : idx * args.num_samples]
                if _use_curated_trace_mode(args):
                    grouped_prm_results = prm_results_by_name[spec_key][(idx - 1) * 2 : idx * 2]
                    grouped_prm_debug = prm_debug_by_name[spec_key][(idx - 1) * 2 : idx * 2]
                elif _use_inline_solution_mode(args):
                    grouped_prm_results = prm_results_by_name[spec_key]
                    grouped_prm_debug = prm_debug_by_name[spec_key]
                elif args.manual_two_traces:
                    grouped_prm_results = prm_results_by_name[spec_key]
                    grouped_prm_debug = prm_debug_by_name[spec_key]
                best_index, best_response, best_prm_result = best_selection_by_name[spec_key][idx - 1]
                reward_field = reward_fields_by_name[spec_key]
                print(f"PRM selection: {spec['label']} using {reward_field}")
                print(f"  selected candidate: {best_index + 1}/{candidate_count}")
                print(f"  selected score: {_select_reward_value(best_prm_result, reward_field):.6f}")
                print()
                for candidate_idx, (response, prm_result, prm_debug) in enumerate(
                    zip(candidate_responses, grouped_prm_results, grouped_prm_debug),
                    start=1,
                ):
                    if _use_curated_trace_mode(args):
                        candidate_label = "Correct trace" if candidate_idx == 1 else "Wrong trace"
                    elif _use_inline_solution_mode(args):
                        candidate_label = f"Inline solution {candidate_idx}/{candidate_count}"
                    elif args.manual_two_traces:
                        candidate_label = "Correct trace" if candidate_idx == 1 else "Wrong trace"
                    else:
                        candidate_label = f"Candidate {candidate_idx}/{candidate_count}"
                    steps = _split_reasoning_steps(response)
                    print(candidate_label)
                    print("Sampled generation:")
                    print(response)
                    print()
                    print("Separated steps:")
                    for step_idx, step in enumerate(steps, start=1):
                        print(f"  Step {step_idx}: {step}")
                    print()
                    if args.debug_prm or args.compare_class_indices:
                        _print_prm_debug_block(spec, prm_result, prm_debug, args.compare_class_indices)
                    else:
                        _print_prm_block(spec, prm_result)
                print("Best-of-N selected solution:")
                print(best_response)
                print()


if __name__ == "__main__":
    main()