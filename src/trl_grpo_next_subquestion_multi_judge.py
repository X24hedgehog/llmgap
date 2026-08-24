#!/usr/bin/env python3
from __future__ import annotations

"""
TRL GRPO for next-subquestion generation with multiple specialized small judges.

Main idea:
- Instead of one big judge checking semantic equivalence, use several small judges.
- Each judge has a specific prompt and checks one criterion only.
- This follows the supervisor's suggestion more directly:
  many cheap judges, each checking an easier subtask.

Reward components:
1. question format reward, rule-based
2. concise / no-solution reward, rule-based
3. lexical overlap reward, rule-based
4. number overlap reward, rule-based
5. quantity-type judge
6. entity judge
7. operation judge
8. next-step judge
9. semantic-equivalence judge

Recommended smoke test:

python src/trl_grpo_next_subquestion_multi_judge.py \
  --out-dir /cluster/scratch/$USER/llmgap/output/trl_ns_multi_judge_qwen3b_50 \
  --policy-model-name Qwen/Qwen2.5-3B-Instruct \
  --judge-model-name Qwen/Qwen2.5-1.5B-Instruct \
  --max-rows 50 \
  --train-mode lora \
  --policy-load-in-4bit \
  --judge-load-in-4bit \
  --num-generations 2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --max-prompt-length 1024 \
  --max-completion-length 64 \
  --lr 5e-6 \
  --epochs 1 \
  --attn-implementation eager \
  --logging-steps 1 \
  --save-steps 25 \
  --debug-print-rewards 20

If Qwen 1.5B judge is still too lenient, try:

  --judge-model-name Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import inspect
import os
import re
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


DEFAULT_NEXT_SUBQUESTION_CSV = (
    "/cluster/home/tunguyen1/llmgap/reasoning-efficiency/"
    "experiments/proof_search/out/next_subquestion_pairs_gsm8k.csv"
)

DEFAULT_POLICY_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


QUESTION_STARTS = (
    "what",
    "how",
    "how many",
    "how much",
    "which",
    "when",
    "where",
    "who",
    "why",
)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "does", "do", "did", "of", "to", "in", "on", "for", "with",
    "and", "or", "by", "from", "as", "at", "into", "than",
    "how", "many", "much", "what", "which", "long", "would",
    "could", "should", "we", "you", "they", "he", "she", "it",
    "his", "her", "their", "its", "total", "number", "amount",
}

BAD_SOLUTION_MARKERS = [
    "therefore",
    "the answer is",
    "answer:",
    "####",
    "step 1",
    "step-by-step",
    "let's solve",
    "solve this",
    "we calculate",
    "we get",
    "=",
]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------
def make_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
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


def clean_generation(text: str) -> str:
    text = str(text).strip()

    prefixes = [
        "Next subquestion:",
        "Subquestion:",
        "Question:",
        "The next subquestion is:",
    ]

    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        text = lines[0].strip()

    qpos = text.find("?")
    if qpos >= 0:
        text = text[: qpos + 1].strip()

    return text


def content_tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[A-Za-z0-9]+", str(text).lower())
        if t not in STOPWORDS and len(t) > 1
    }


def numbers_and_fractions(text: str) -> set[str]:
    text = str(text).lower()

    fracs = set(re.findall(r"\d+\s*/\s*\d+", text))
    fracs = {f.replace(" ", "") for f in fracs}

    nums = set(re.findall(r"\d+(?:\.\d+)?", text))

    word_fracs = {
        "half": "1/2",
        "one half": "1/2",
        "third": "1/3",
        "one third": "1/3",
        "quarter": "1/4",
        "one quarter": "1/4",
        "fourth": "1/4",
        "one fourth": "1/4",
    }

    for phrase, frac in word_fracs.items():
        if phrase in text:
            fracs.add(frac)

    return nums | fracs


def parse_yes_no(text: str) -> bool:
    text = str(text).strip().lower()

    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False

    # fallback
    first_word = re.findall(r"[a-z]+", text)
    if first_word:
        return first_word[0] == "yes"

    return False


# ---------------------------------------------------------------------
# Rule-based reward components
# ---------------------------------------------------------------------
def question_likeness_score(pred: str) -> float:
    text = clean_generation(pred).strip().lower()

    if not text:
        return 0.0

    starts_like_question = text.startswith(QUESTION_STARTS)
    ends_question = text.endswith("?")

    if starts_like_question and ends_question:
        return 1.0
    if ends_question:
        return 0.6
    if starts_like_question:
        return 0.4
    return 0.0


def concise_no_solution_score(pred: str) -> float:
    text = clean_generation(pred).strip()
    lower = text.lower()

    if not text:
        return 0.0

    if any(marker in lower for marker in BAD_SOLUTION_MARKERS):
        return 0.0

    n_words = len(text.split())

    if 4 <= n_words <= 25:
        return 1.0
    if 26 <= n_words <= 40:
        return 0.5
    if 1 <= n_words < 4:
        return 0.3

    return 0.0


def lexical_overlap_score(pred: str, gold: str) -> float:
    pred_tokens = content_tokens(clean_generation(pred))
    gold_tokens = content_tokens(gold)

    if not gold_tokens:
        return 0.0

    overlap = len(pred_tokens & gold_tokens) / len(gold_tokens)

    if overlap >= 0.70:
        return 1.0
    if overlap >= 0.40:
        return 0.6
    if overlap > 0:
        return 0.3

    return 0.0


def number_overlap_score(pred: str, gold: str) -> float:
    pred_nums = numbers_and_fractions(clean_generation(pred))
    gold_nums = numbers_and_fractions(gold)

    if not gold_nums:
        return 1.0

    if gold_nums <= pred_nums:
        return 1.0

    if gold_nums & pred_nums:
        return 0.5

    return 0.0


# ---------------------------------------------------------------------
# Specialized judge prompts
# ---------------------------------------------------------------------
def quantity_type_prompt(context: str, gold: str, pred: str) -> str:
    return f"""You are a strict verifier. Your only job is to check QUANTITY TYPE.

Question:
Do the gold subquestion and predicted subquestion ask for the same kind of quantity?

Answer YES only if both ask for the same type of value, such as:
- money / cost / price
- time / minutes / hours
- count / number of objects
- distance / length
- weight
- rate / per-unit amount
- total amount
- remaining amount
- difference between two amounts

Answer NO if:
- one asks for a rate and the other asks for a total
- one asks for a cost and the other asks for a count
- one asks for time and the other asks for number of cycles
- one asks for an intermediate quantity and the other asks for a final total of a different type

Examples:

Gold: How many points does Martha earn for every dollar spent?
Predicted: How many points does Martha get from her total spending, including the bonus?
Answer: NO
Reason: Gold asks for a rate, predicted asks for total points.

Gold: What is the cost of each elliptical machine?
Predicted: How many total cardio machines does each gym need to replace?
Answer: NO
Reason: Gold asks for cost, predicted asks for count.

Gold: How many minutes are there in 2 hours?
Predicted: How long is 2 hours in minutes?
Answer: YES
Reason: Both ask for time in minutes.

Gold: How many blankets did Freddie and his team collect on the second day?
Predicted: How many blankets did they collect on day two?
Answer: YES
Reason: Both ask for a count of blankets.

Now judge this case.

Problem context:
{context}

Gold subquestion:
{gold}

Predicted subquestion:
{pred}

Answer only YES or NO."""


def entities_prompt(context: str, gold: str, pred: str) -> str:
    return f"""You are a strict verifier. Your only job is to check MAIN ENTITIES AND OBJECTS.

Question:
Do the gold subquestion and predicted subquestion refer to the same main people, objects, and attributes?

Answer YES if:
- names or pronouns refer to the same person
- objects are the same or clearly equivalent
- wording differs but the entity/object is the same

Answer NO if:
- the gold asks about one object but the prediction asks about a different object
- the gold asks about one person/group but the prediction asks about another
- the prediction is too vague and loses the important entity/object
- the prediction asks about the whole problem instead of the specific object in the gold

Examples:

Gold: How many minutes does Marcy spend combing her cat?
Predicted: How long does Marcy spend brushing the cat?
Answer: YES
Reason: Marcy and the cat are the same entities.

Gold: What is the cost of each elliptical machine?
Predicted: What is the cost of each treadmill?
Answer: NO
Reason: Elliptical machine and treadmill are different objects.

Gold: How many trolls are hiding in the plains?
Predicted: How many trolls are hiding under the bridge?
Answer: NO
Reason: Plains and bridge are different locations.

Gold: How many blankets did Freddie and his team collect on the second day?
Predicted: How many blankets did they collect on the second day?
Answer: YES
Reason: "they" refers to Freddie and his team.

Now judge this case.

Problem context:
{context}

Gold subquestion:
{gold}

Predicted subquestion:
{pred}

Answer only YES or NO."""


def operation_prompt(context: str, gold: str, pred: str) -> str:
    return f"""You are a strict verifier. Your only job is to check IMPLIED OPERATION.

Question:
Do the gold subquestion and predicted subquestion require the same arithmetic or logical operation?

Answer YES if both require the same calculation, such as:
- the same multiplication
- the same division
- the same addition
- the same subtraction
- the same unit conversion
- the same percentage/rate calculation

Answer NO if:
- one asks for a conversion and the other asks for a division
- one asks for a rate and the other asks for applying that rate
- one asks for an intermediate value and the other asks for a later total
- one asks for a count and the other asks for a cost
- they are merely related to the same problem but require different calculations

Examples:

Gold: How many minutes are there in 2 hours?
Predicted: How many complete cycles can Barry fit into a 2-hour period?
Answer: NO
Reason: Gold requires converting hours to minutes; prediction requires dividing total time by cycle length.

Gold: What is the cost of each elliptical machine?
Predicted: What is twice the cost of a treadmill?
Answer: YES
Reason: Both require multiplying the treadmill cost by 2.

Gold: How many points does Martha earn for every dollar spent?
Predicted: How many points does Martha earn for each dollar?
Answer: YES
Reason: Both ask for the same per-dollar rate.

Gold: How many points does Martha earn for every dollar spent?
Predicted: How many total points does Martha earn including the bonus?
Answer: NO
Reason: Gold asks for the rate; prediction asks for applying the rate and adding bonus.

Now judge this case.

Problem context:
{context}

Gold subquestion:
{gold}

Predicted subquestion:
{pred}

Answer only YES or NO."""


def next_step_prompt(context: str, gold: str, pred: str) -> str:
    return f"""You are a strict verifier. Your only job is to check SAME NEXT REASONING STEP.

Question:
Does the predicted subquestion ask for the same next intermediate quantity as the gold subquestion?

Answer YES only if:
- the predicted question is at the same point in the reasoning chain as the gold
- it asks for the same next intermediate quantity
- it does not jump ahead to the final answer
- it does not go back to an already-solved earlier quantity

Answer NO if:
- it asks for a later quantity
- it asks for the final answer too early
- it asks for an earlier quantity that is already known
- it is related to the problem but not the same next step

Examples:

Problem context: Barry stands on his head for 10 minutes, then sits for 5 minutes. We already know one complete cycle takes 15 minutes. The final question asks how many turns fit in 2 hours.
Gold: How many minutes are there in 2 hours?
Predicted: How many complete cycles can Barry fit into a 2-hour period?
Answer: NO
Reason: The prediction jumps to the later division step; the gold asks for the next conversion step.

Problem context: The treadmill cost is already known. Ellipticals cost twice as much as treadmills.
Gold: What is the cost of each elliptical machine?
Predicted: What is twice the cost of each treadmill?
Answer: YES
Reason: Both ask for the same next intermediate quantity.

Problem context: We already know the total amount Martha spent. She earns 50 points per $10.
Gold: How many points does Martha earn for every dollar spent?
Predicted: How many total points does Martha earn including the bonus?
Answer: NO
Reason: The prediction jumps to a later/final quantity.

Gold: How many blankets did Freddie and his team collect on the second day?
Predicted: How many blankets did they collect on the second day?
Answer: YES
Reason: Same next intermediate quantity.

Now judge this case.

Problem context:
{context}

Gold subquestion:
{gold}

Predicted subquestion:
{pred}

Answer only YES or NO."""


def semantic_equivalence_prompt(context: str, gold: str, pred: str) -> str:
    return f"""You are a strict verifier. Your only job is to check FULL SEMANTIC EQUIVALENCE.

Question:
Do the gold subquestion and predicted subquestion ask for exactly the same unknown quantity?

Answer YES only if:
- the two questions would have the same answer
- the two questions ask for the same quantity
- wording differences do not change the meaning

Answer NO if:
- the questions are only related but not equivalent
- one asks for a rate and the other asks for a total
- one asks for an intermediate step and the other asks for a final answer
- one asks for a different object, person, time, or unit
- one question is more general or more vague than the other
- both questions are useful for the problem, but they ask for different unknowns

Examples:

Gold: How many minutes does Marcy spend combing her cat?
Predicted: How long does Marcy spend brushing the cat?
Answer: YES
Reason: Same unknown quantity, different wording.

Gold: How many minutes are there in 2 hours?
Predicted: How many complete cycles can Barry fit into a 2-hour period?
Answer: NO
Reason: Different unknown quantities.

Gold: What is the cost of each elliptical machine?
Predicted: How many total cardio machines does each gym need to replace?
Answer: NO
Reason: Cost and count are different unknown quantities.

Gold: How many points does Martha earn for every dollar spent?
Predicted: How many points does Martha get from her total spending, including the bonus?
Answer: NO
Reason: Rate and total are different unknown quantities.

Gold: How many blankets did Freddie and his team collect on the second day?
Predicted: How many blankets did they collect on day two?
Answer: YES
Reason: Same unknown quantity.

Now judge this case.

Problem context:
{context}

Gold subquestion:
{gold}

Predicted subquestion:
{pred}

Answer only YES or NO."""


# ---------------------------------------------------------------------
# Specialized judge class
# ---------------------------------------------------------------------
class MultiSpecializedJudge:
    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool,
        attn_implementation: str,
        max_new_tokens: int,
        batch_size: int,
        enabled_judges: list[str],
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.enabled_judges = enabled_judges

        self.prompt_builders = {
            "quantity_type": quantity_type_prompt,
            "entities": entities_prompt,
            "operation": operation_prompt,
            "next_step": next_step_prompt,
            "semantic_equivalence": semantic_equivalence_prompt,
        }

        print("=" * 80, flush=True)
        print("Loading multi specialized judge", flush=True)
        print(f"judge_model_name: {model_name}", flush=True)
        print(f"judge_load_in_4bit: {load_in_4bit}", flush=True)
        print(f"enabled_judges: {enabled_judges}", flush=True)
        print("=" * 80, flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=make_quant_config(load_in_4bit),
            torch_dtype=None if load_in_4bit else torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def render_chat(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def run_yes_no_prompts(self, prompts: list[str]) -> list[bool]:
        rendered = [self.render_chat(p) for p in prompts]
        results: list[bool] = []

        for start in range(0, len(rendered), self.batch_size):
            batch_texts = rendered[start:start + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            decoded = self.tokenizer.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
            )

            results.extend([parse_yes_no(x) for x in decoded])

        return results

    def judge_one_criterion(
        self,
        criterion: str,
        contexts: list[str],
        golds: list[str],
        preds: list[str],
    ) -> list[bool]:
        builder = self.prompt_builders[criterion]

        prompts = [
            builder(context=c, gold=g, pred=p)
            for c, g, p in zip(contexts, golds, preds)
        ]

        return self.run_yes_no_prompts(prompts)

    def judge_batch(
        self,
        contexts: list[str],
        golds: list[str],
        preds: list[str],
    ) -> dict[str, list[bool]]:
        outputs: dict[str, list[bool]] = {}

        for criterion in self.enabled_judges:
            outputs[criterion] = self.judge_one_criterion(
                criterion=criterion,
                contexts=contexts,
                golds=golds,
                preds=preds,
            )

        return outputs


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
def apply_policy_chat_template(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_policy_prompt(raw_prompt: str) -> str:
    return f"""Given the following math problem and reasoning context, write only the next useful subquestion.

Rules:
- Output exactly one question.
- Do not answer the question.
- Do not explain.
- Do not include reasoning steps.
- End with a question mark.

Context:
{str(raw_prompt).strip()}

Next subquestion:"""


def build_dataset(args, tokenizer) -> Dataset:
    df = pd.read_csv(args.train_csv)

    required = {args.prompt_col, args.target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")

    if args.split_col and args.split_col in df.columns:
        df = df[
            df[args.split_col].astype(str).str.lower() == args.train_split.lower()
        ].reset_index(drop=True)

    valid = df[args.prompt_col].notna() & df[args.target_col].notna()
    if int((~valid).sum()):
        print(f"WARNING: dropping {int((~valid).sum())} rows with NaN prompt/target", flush=True)
    df = df[valid].reset_index(drop=True)

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].reset_index(drop=True)

    raw_contexts = df[args.prompt_col].astype(str).tolist()
    golds = df[args.target_col].astype(str).tolist()

    policy_prompts = [
        apply_policy_chat_template(
            tokenizer=tokenizer,
            prompt=build_policy_prompt(context),
            use_chat_template=not args.no_chat_template,
        )
        for context in raw_contexts
    ]

    out = pd.DataFrame()
    out["prompt"] = policy_prompts
    out["context"] = raw_contexts
    out["gold_subquestion"] = golds

    print("=" * 80, flush=True)
    print("Loaded next-subquestion dataset", flush=True)
    print(f"csv: {args.train_csv}", flush=True)
    print(f"rows: {len(out)}", flush=True)
    print(f"prompt_col: {args.prompt_col}", flush=True)
    print(f"target_col: {args.target_col}", flush=True)
    print(f"train_split: {args.train_split}", flush=True)
    print(f"chat_template: {not args.no_chat_template}", flush=True)

    if len(out):
        print("\nExample policy prompt:", flush=True)
        print(out.iloc[0]["prompt"][:2000], flush=True)
        print("\nExample gold_subquestion:", flush=True)
        print(out.iloc[0]["gold_subquestion"], flush=True)

    print("=" * 80, flush=True)

    return Dataset.from_pandas(out, preserve_index=False)


# ---------------------------------------------------------------------
# Policy model / training config
# ---------------------------------------------------------------------
def load_policy_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


def load_policy_model(args, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        args.policy_model_name,
        quantization_config=make_quant_config(args.policy_load_in_4bit),
        torch_dtype=None if args.policy_load_in_4bit else torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False

    if args.train_mode == "head":
        configure_head_only(model, args.train_embed_too)

    return model


def configure_head_only(model, train_embed_too: bool = False):
    for _, p in model.named_parameters():
        p.requires_grad = False

    trainable = []

    if hasattr(model, "lm_head") and model.lm_head is not None:
        for p in model.lm_head.parameters():
            p.requires_grad = True
        trainable.append("lm_head")

    if train_embed_too:
        input_emb = model.get_input_embeddings()
        if input_emb is not None:
            for p in input_emb.parameters():
                p.requires_grad = True
            trainable.append("input_embeddings")

    total = sum(p.numel() for p in model.parameters())
    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 80, flush=True)
    print("HEAD-ONLY TRAINING", flush=True)
    print("Trainable modules:", trainable, flush=True)
    print(f"Trainable params: {trainable_n:,} / {total:,} ({100 * trainable_n / max(total, 1):.6f}%)", flush=True)
    print("=" * 80, flush=True)


def build_lora_config(args):
    if args.train_mode != "lora":
        return None

    if LoraConfig is None:
        raise ImportError("peft is not installed, but --train-mode lora was selected.")

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
        bias="none",
    )


def build_grpo_config(args) -> GRPOConfig:
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
        "max_grad_norm": args.max_grad_norm,
    }

    sig = inspect.signature(GRPOConfig.__init__)
    accepted = set(sig.parameters.keys())

    filtered = {k: v for k, v in desired_kwargs.items() if k in accepted}
    dropped = {k: v for k, v in desired_kwargs.items() if k not in accepted}

    print("\nGRPOConfig accepted keys:", sorted(filtered.keys()), flush=True)
    if dropped:
        print("GRPOConfig dropped unsupported keys:", sorted(dropped.keys()), flush=True)

    config = GRPOConfig(**filtered)

    for field in ["num_generations", "max_prompt_length", "max_completion_length", "temperature", "top_p", "beta"]:
        if hasattr(config, field):
            setattr(config, field, getattr(args, field))
            print(f"Set GRPOConfig.{field} = {getattr(args, field)}", flush=True)

    return config


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("TRL GRPO next-subquestion with multiple specialized judges")

    p.add_argument("--policy-model-name", default=DEFAULT_POLICY_MODEL)
    p.add_argument("--judge-model-name", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--train-csv", default=DEFAULT_NEXT_SUBQUESTION_CSV)
    p.add_argument("--out-dir", required=True)

    p.add_argument("--prompt-col", default="prompt")
    p.add_argument("--target-col", default="next_subquestion")
    p.add_argument("--split-col", default="split")
    p.add_argument("--train-split", default="train")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train-mode", choices=["lora", "head", "full"], default="lora")
    p.add_argument("--policy-load-in-4bit", action="store_true")
    p.add_argument("--judge-load-in-4bit", action="store_true")
    p.add_argument("--train-embed-too", action="store_true")

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    p.add_argument("--num-generations", type=int, default=2)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-6)

    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-completion-length", type=int, default=64)

    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.1)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--attn-implementation", default="eager")

    p.add_argument("--judge-max-new-tokens", type=int, default=4)
    p.add_argument("--judge-batch-size", type=int, default=8)

    p.add_argument(
        "--enabled-judges",
        default="quantity_type,entities,operation,next_step,semantic_equivalence",
        help="Comma-separated judge names.",
    )

    p.add_argument("--question-weight", type=float, default=0.10)
    p.add_argument("--concise-weight", type=float, default=0.10)
    p.add_argument("--lexical-weight", type=float, default=0.10)
    p.add_argument("--number-weight", type=float, default=0.10)

    p.add_argument("--quantity-type-weight", type=float, default=0.10)
    p.add_argument("--entities-weight", type=float, default=0.10)
    p.add_argument("--operation-weight", type=float, default=0.15)
    p.add_argument("--next-step-weight", type=float, default=0.25)
    p.add_argument("--semantic-equivalence-weight", type=float, default=0.30)

    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--debug-print-rewards", type=int, default=20)

    args = p.parse_args()

    if args.per_device_train_batch_size % args.num_generations != 0:
        raise ValueError(
            "per-device-train-batch-size must be divisible by num-generations. "
            f"Got batch={args.per_device_train_batch_size}, generations={args.num_generations}."
        )

    args.enabled_judges = [
        x.strip()
        for x in args.enabled_judges.split(",")
        if x.strip()
    ]

    allowed = {
        "quantity_type",
        "entities",
        "operation",
        "next_step",
        "semantic_equivalence",
    }

    bad = set(args.enabled_judges) - allowed
    if bad:
        raise ValueError(f"Unknown judges: {bad}. Allowed: {allowed}")

    return args


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    policy_tokenizer = load_policy_tokenizer(args)
    dataset = build_dataset(args, policy_tokenizer)

    policy_model = load_policy_model(args, policy_tokenizer)
    peft_config = build_lora_config(args)
    training_args = build_grpo_config(args)

    judge = MultiSpecializedJudge(
        model_name=args.judge_model_name,
        load_in_4bit=args.judge_load_in_4bit,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.judge_max_new_tokens,
        batch_size=args.judge_batch_size,
        enabled_judges=args.enabled_judges,
    )

    judge_weights = {
        "quantity_type": args.quantity_type_weight,
        "entities": args.entities_weight,
        "operation": args.operation_weight,
        "next_step": args.next_step_weight,
        "semantic_equivalence": args.semantic_equivalence_weight,
    }

    print("=" * 80, flush=True)
    print("TRAINING CONFIG", flush=True)
    print(f"policy_model_name: {args.policy_model_name}", flush=True)
    print(f"judge_model_name: {args.judge_model_name}", flush=True)
    print(f"enabled_judges: {args.enabled_judges}", flush=True)
    print(f"judge_weights: {judge_weights}", flush=True)
    print(f"train_mode: {args.train_mode}", flush=True)
    print(f"policy_load_in_4bit: {args.policy_load_in_4bit}", flush=True)
    print(f"judge_load_in_4bit: {args.judge_load_in_4bit}", flush=True)
    print(f"num_generations: {args.num_generations}", flush=True)
    print(f"batch_size: {args.per_device_train_batch_size}", flush=True)
    print(f"max_completion_length: {args.max_completion_length}", flush=True)
    print("=" * 80, flush=True)

    debug = {"n": 0}

    def question_reward_fn(completions, **kwargs):
        preds = [completion_to_text(c) for c in completions]
        return [args.question_weight * question_likeness_score(p) for p in preds]

    def concise_reward_fn(completions, **kwargs):
        preds = [completion_to_text(c) for c in completions]
        return [args.concise_weight * concise_no_solution_score(p) for p in preds]

    def lexical_reward_fn(completions, gold_subquestion=None, **kwargs):
        if gold_subquestion is None:
            raise ValueError("Missing `gold_subquestion` from dataset.")
        preds = [completion_to_text(c) for c in completions]
        return [
            args.lexical_weight * lexical_overlap_score(p, g)
            for p, g in zip(preds, gold_subquestion)
        ]

    def number_reward_fn(completions, gold_subquestion=None, **kwargs):
        if gold_subquestion is None:
            raise ValueError("Missing `gold_subquestion` from dataset.")
        preds = [completion_to_text(c) for c in completions]
        return [
            args.number_weight * number_overlap_score(p, g)
            for p, g in zip(preds, gold_subquestion)
        ]

    def multi_judge_reward_fn(completions, context=None, gold_subquestion=None, **kwargs):
        if context is None or gold_subquestion is None:
            raise ValueError("Missing `context` or `gold_subquestion` from dataset.")

        preds_raw = [completion_to_text(c) for c in completions]
        preds = [clean_generation(p) for p in preds_raw]
        contexts = [str(c) for c in context]
        golds = [str(g) for g in gold_subquestion]

        judge_outputs = judge.judge_batch(
            contexts=contexts,
            golds=golds,
            preds=preds,
        )

        n = len(preds)
        rewards = [0.0 for _ in range(n)]

        for criterion, values in judge_outputs.items():
            w = judge_weights[criterion]
            for i, v in enumerate(values):
                rewards[i] += w * float(v)

        if args.debug_print_rewards > 0 and debug["n"] < args.debug_print_rewards:
            for i in range(n):
                if debug["n"] >= args.debug_print_rewards:
                    break

                raw_pred = preds_raw[i]
                pred = preds[i]
                gold = golds[i]
                ctx = contexts[i]

                q_score = question_likeness_score(raw_pred)
                c_score = concise_no_solution_score(raw_pred)
                lex_score = lexical_overlap_score(raw_pred, gold)
                num_score = number_overlap_score(raw_pred, gold)

                rule_reward = (
                    args.question_weight * q_score
                    + args.concise_weight * c_score
                    + args.lexical_weight * lex_score
                    + args.number_weight * num_score
                )

                judge_result_one = {
                    criterion: judge_outputs[criterion][i]
                    for criterion in args.enabled_judges
                }

                total_reward = rule_reward + rewards[i]

                print("=" * 80, flush=True)
                print("CONTEXT:", flush=True)
                print(ctx[:2000], flush=True)
                print("\nGOLD_SUBQUESTION:", flush=True)
                print(gold, flush=True)
                print("\nRAW_COMPLETION:", flush=True)
                print(raw_pred, flush=True)
                print("\nCLEAN_PRED:", flush=True)
                print(pred, flush=True)
                print("\nRULE SCORES:", flush=True)
                print(f"question_likeness: {q_score} -> {args.question_weight * q_score}", flush=True)
                print(f"concise_no_solution: {c_score} -> {args.concise_weight * c_score}", flush=True)
                print(f"lexical_overlap: {lex_score} -> {args.lexical_weight * lex_score}", flush=True)
                print(f"number_overlap: {num_score} -> {args.number_weight * num_score}", flush=True)
                print("\nSPECIALIZED JUDGE OUTPUTS:", flush=True)
                print(judge_result_one, flush=True)
                print(f"multi_judge_reward: {rewards[i]}", flush=True)
                print(f"rule_reward: {rule_reward}", flush=True)
                print(f"TOTAL_REWARD: {total_reward}", flush=True)
                print("=" * 80, flush=True)

                debug["n"] += 1

        return rewards

    reward_funcs = [
        question_reward_fn,
        concise_reward_fn,
        lexical_reward_fn,
        number_reward_fn,
        multi_judge_reward_fn,
    ]

    trainer_kwargs = dict(
        model=policy_model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=policy_tokenizer,
    )

    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    try:
        trainer = GRPOTrainer(**trainer_kwargs)
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise

        print("GRPOTrainer does not accept processing_class; falling back to tokenizer=tokenizer", flush=True)
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = policy_tokenizer
        trainer = GRPOTrainer(**trainer_kwargs)

    print(
        "trainer.args.max_completion_length actual = "
        f"{getattr(trainer.args, 'max_completion_length', None)}",
        flush=True,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    policy_tokenizer.save_pretrained(args.out_dir)

    print(f"Saved model/adapters to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()