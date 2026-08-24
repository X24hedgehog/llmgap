#!/usr/bin/env python3
"""Unified fine-tuning script for all tasks: correct_answer, next_subquestion, distractor.

Works across experimental settings:
  - colm-paper-code-cleaned (correct_answer + distractor)
  - reasoning-efficiency (correct_answer + next_subquestion)

Key features:
  - 80/20 train/val split from training data
  - Per-epoch checkpoint saving
  - Post-training judge-based evaluation of all checkpoints on val set
  - Auto-selects best checkpoint by validation accuracy
  - LoRA r=64, alpha=128
  - Saves val_accuracy.json per checkpoint for run_inference.py to find best

Example:
    python finetune.py \\
        --model-name Qwen/Qwen2.5-0.5B-Instruct \\
        --task correct_answer \\
        --train-csv path/to/correct_answer_pairs.csv \\
        --out-dir out/checkpoints/Qwen2.5-0.5B-Instruct_correct_answer
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openai_judge import DEFAULT_OPENAI_EEDI_JUDGE_MODEL, judge_yes_no_openai
from prompt import (
    build_answer_equivalence_prompt,
)

try:
    from trl import SFTConfig as _SFTConfigCls
    _USE_SFT_CONFIG = True
except ImportError:
    from transformers import TrainingArguments as _SFTConfigCls
    _USE_SFT_CONFIG = False

# ── constants ─────────────────────────────────────────────────────────────────

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LARGE_MODELS = {"7b", "8b"}
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

CORRECT_ANSWER_INSTRUCTION = "\n\nSolve this problem step by step."

# Simplified distractor prompts: ask for 1 distractor (not full set)
DISTRACTOR_PROMPT_BY_TYPE = {
    "ContCompContMisconceptionIncons": (
        "You are an expert math educator generating a distractor based on "
        "the Comparison Inconsistency misconception.\n\n"
        "When a problem says \"A has N more than B\", the correct way to find B "
        "is B = A - N. A student with this misconception FLIPS the operation: "
        "they compute B = A + N instead. Similarly, \"fewer than\" → they subtract "
        "instead of add, \"times as many\" → they multiply instead of "
        "divide, etc.\n\n"
        "Your job is to simulate this mistake and produce a single distractor "
        "answer for the following math question.\n\n"
        "Question: {problem}\n"
        "Correct answer: {correct_answer}\n\n"
        "Think step by step, flipping some comparison operations to arrive at "
        "a wrong answer."
    ),
    "ContTransferContMisconceptionIncons": (
        "You are an expert math educator generating a distractor based on "
        "the Transfer Inconsistency misconception.\n\n"
        "When a problem says \"A gives B N items\", the correct effect on A "
        "is: A loses N (subtract). A student with this misconception FLIPS "
        "the operation: they add instead of subtract (or vice versa), "
        "confusing the direction of the transfer.\n\n"
        "Your job is to simulate this mistake and produce a single distractor "
        "answer for the following math question.\n\n"
        "Question: {problem}\n"
        "Correct answer: {correct_answer}\n\n"
        "Think step by step, flipping some transfer operations to arrive at "
        "a wrong answer."
    ),
}

TASK_MAX_NEW_TOKENS = {
    "correct_answer": 512,
    "next_subquestion": 80,
    "distractor": 512,
}

EEDI_PROMPT_STYLES = {"eedi_correct_answer", "eedi_distractor"}


def _needs_4bit(model_name: str) -> bool:
    return any(tag in model_name.lower() for tag in LARGE_MODELS)


# ── prompt & response formatting ──────────────────────────────────────────────

def _format_chat(tokenizer, prompt: str, response: str) -> str:
    return f"{prompt.rstrip()}\n\n{str(response).strip()}"


def _build_prompt(row: pd.Series, task: str) -> str:
    prompt_style = str(row.get("prompt_style", ""))
    if prompt_style == "eedi_correct_answer":
        return f"Question: {str(row['question']).strip()}" + CORRECT_ANSWER_INSTRUCTION
    if prompt_style == "eedi_distractor":
        return (
            "You are solving a math question as a student with the following "
            f"misconception: {str(row['misconception_name']).strip()}\n\n"
            f"Question: {str(row['question']).strip()}\n\n"
            "Think step by step and give the incorrect answer this student "
            "would produce."
        )

    if task == "correct_answer":
        return str(row["prompt"]).rstrip() + CORRECT_ANSWER_INSTRUCTION
    elif task == "next_subquestion":
        return str(row["prompt"])
    elif task == "distractor":
        mtype = row["misconception_type"]
        template = DISTRACTOR_PROMPT_BY_TYPE[mtype]
        return template.format(
            problem=row["problem"],
            correct_answer=row["correct_answer"],
        )
    raise ValueError(f"Unknown task: {task}")


def _build_response(row: pd.Series, task: str) -> str:
    if task == "correct_answer":
        return str(row["target_question_reasoning_trace"])
    elif task == "next_subquestion":
        return str(row["next_subquestion"])
    elif task == "distractor":
        # Pick 1 random explained trace
        traces = json.loads(row["target_distractor_explained_traces"])
        return random.choice(traces)
    raise ValueError(f"Unknown task: {task}")


# ── dataset builder ───────────────────────────────────────────────────────────

def build_hf_dataset(df: pd.DataFrame, tokenizer, task: str) -> Dataset:
    texts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Formatting dataset"):
        prompt = _build_prompt(row, task)
        response = _build_response(row, task)
        texts.append(_format_chat(tokenizer, prompt, response))
    return Dataset.from_dict({"text": texts})


def build_val_data(df: pd.DataFrame, task: str):
    """Build validation prompts and gold answers for judge-based scoring."""
    prompts, golds, prompt_styles = [], [], []
    for _, row in df.iterrows():
        prompts.append(_build_prompt(row, task))
        prompt_styles.append(str(row.get("prompt_style", "")))
        if task == "correct_answer":
            golds.append(str(row["target_answer"]))
        elif task == "next_subquestion":
            golds.append(str(row["next_subquestion"]))
        elif task == "distractor":
            golds.append(row["target_distractor_answers"])  # JSON string
    return prompts, golds, prompt_styles


# ── post-training checkpoint evaluation (judge-based) ─────────────────────────

def _generate_predictions(model, tokenizer, prompts, task, batch_size=4):
    """Generate predictions for all validation prompts."""
    max_new_tokens = TASK_MAX_NEW_TOKENS.get(task, 256)
    outputs = []
    model.eval()
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating val preds"):
        batch = prompts[i : i + batch_size]
        formatted = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
        ]
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for seq in out:
            text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            outputs.append(text.strip())
    return outputs


def _load_judge():
    """Load the Qwen 7B judge model and tokenizer."""
    print(f"Loading judge model: {JUDGE_MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def _judge_yes_no(judge_model, judge_tok, prompts, batch_size=8):
    """Run yes/no judge prompts and return list of 0/1 scores."""
    scores = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="LLM judge"):
        batch = prompts[i : i + batch_size]
        formatted = [
            judge_tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
        ]
        enc = judge_tok(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(judge_model.device)
        with torch.no_grad():
            out = judge_model.generate(
                **enc,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=judge_tok.pad_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        for seq in out:
            text = judge_tok.decode(
                seq[input_len:], skip_special_tokens=True
            ).strip()
            verdicts = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
            final_verdict = verdicts[-1].lower() if verdicts else "no"
            scores.append(1 if final_verdict == "yes" else 0)
    return scores


def _parse_distractor_gold(gold_str: str):
    import ast

    parsed = ast.literal_eval(gold_str)
    return parsed if isinstance(parsed, list) else [parsed]


def _use_semantic_distractor_scoring(prompt_style: str) -> bool:
    return str(prompt_style) == "eedi_distractor"


def _use_semantic_correct_answer_scoring(prompt_style: str) -> bool:
    return str(prompt_style) == "eedi_correct_answer"


def _score_distractor_regex(predictions, golds, prompts=None, prompt_styles=None):
    """Score distractor predictions.

    Numeric distractors are matched by final-number extraction, as in the
    synthetic MathGAP setting. EEDI distractors are matched by semantic
    equivalence using the judge model.
    """
    def normalize_text(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"^(distractor\s*\d*\s*:|incorrect student answer:|answer:)\s*", "", text)
        text = text.replace("\\(", "").replace("\\)", "")
        text = text.replace("\\[", "").replace("\\]", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def is_numeric_like(text: str) -> bool:
        text = normalize_text(text)
        return bool(re.fullmatch(r"[-+]?((\d+\.?\d*)|(\.\d+))", text))

    scores = []
    semantic_examples = []
    if prompt_styles is None:
        prompt_styles = [""] * len(predictions)

    for idx, (pred, gold_str, prompt_style) in enumerate(zip(predictions, golds, prompt_styles)):
        gold_values = _parse_distractor_gold(gold_str)

        if not _use_semantic_distractor_scoring(prompt_style):
            gold_nums = {int(float(value)) for value in gold_values}
            all_nums = re.findall(r'-?\b\d+\b', pred)
            if all_nums:
                last_num = int(all_nums[-1])
                scores.append(1 if last_num in gold_nums else 0)
            else:
                scores.append(0)
            continue

        semantic_examples.append((idx, pred, gold_values))
        scores.append(None)

    if not semantic_examples:
        return scores

    if prompts is None:
        raise ValueError("prompts are required for semantic distractor scoring")

    expanded_prompts = []
    row_slices = []
    for row_idx, pred, gold_values in semantic_examples:
        problem_context = prompts[row_idx]
        start_idx = len(expanded_prompts)
        pred_norm = normalize_text(pred)
        for gold in gold_values:
            expanded_prompts.append(
                build_answer_equivalence_prompt(
                    problem_context,
                    pred_norm,
                    normalize_text(gold),
                )
            )
        row_slices.append((row_idx, start_idx, len(expanded_prompts)))

    print(f"Using OpenAI EEDI judge model: {DEFAULT_OPENAI_EEDI_JUDGE_MODEL}")
    expanded_scores = judge_yes_no_openai(expanded_prompts)

    for row_idx, start, end in row_slices:
        scores[row_idx] = 1 if any(expanded_scores[start:end]) else 0

    return scores


def _score_correct_answer(predictions, golds, prompts, prompt_styles, batch_size=8):
    def normalize_text(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"^(answer:|correct answer:)\s*", "", text)
        text = text.replace("\\(", "").replace("\\)", "")
        text = text.replace("\\[", "").replace("\\]", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def is_numeric_like(text: str) -> bool:
        return bool(re.fullmatch(r"[-+]?((\d+\.?\d*)|(\.\d+))", str(text).strip()))

    scores = [None] * len(predictions)
    numeric_examples = []
    semantic_examples = []

    for idx, (prediction, gold, problem_context, prompt_style) in enumerate(
        zip(predictions, golds, prompts, prompt_styles)
    ):
        if _use_semantic_correct_answer_scoring(prompt_style):
            semantic_examples.append((idx, problem_context, prediction, gold))
            continue

        if is_numeric_like(gold):
            numeric_examples.append((idx, prediction, gold))
            continue

        scores[idx] = 1 if normalize_text(prediction) == normalize_text(gold) else 0

    if numeric_examples:
        judge_model, judge_tok = _load_judge()
        numeric_prompts = [
            (
                "A student solved the following math problem and wrote this solution:\n"
                f"{prediction}\n\n"
                f"The correct final answer is: {gold}\n\n"
                "Did the student arrive at the correct final answer? "
                "Answer only 'yes' or 'no'."
            )
            for _, prediction, gold in numeric_examples
        ]
        numeric_scores = _judge_yes_no(judge_model, judge_tok, numeric_prompts, batch_size)
        del judge_model
        torch.cuda.empty_cache()
        for (row_idx, _, _), score in zip(numeric_examples, numeric_scores):
            scores[row_idx] = score

    if semantic_examples:
        semantic_prompts = [
            build_answer_equivalence_prompt(
                problem_context,
                normalize_text(prediction),
                normalize_text(gold),
            )
            for _, problem_context, prediction, gold in semantic_examples
        ]
        print(f"Using OpenAI EEDI judge model: {DEFAULT_OPENAI_EEDI_JUDGE_MODEL}")
        semantic_scores = judge_yes_no_openai(semantic_prompts, batch_size=batch_size)
        for (row_idx, _, _, _), score in zip(semantic_examples, semantic_scores):
            scores[row_idx] = score

    return scores


def _score_correct_answer_exact(predictions, golds):
    """Score non-numeric correct answers by normalized exact match."""

    def normalize_text(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"^(answer:|correct answer:)\s*", "", text)
        text = text.replace("\\(", "").replace("\\)", "")
        text = text.replace("\\[", "").replace("\\]", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    return [
        1 if normalize_text(pred) == normalize_text(gold) else 0
        for pred, gold in zip(predictions, golds)
    ]


def _all_numeric_golds(golds) -> bool:
    def is_numeric_like(text: str) -> bool:
        return bool(re.fullmatch(r"[-+]?((\d+\.?\d*)|(\.\d+))", str(text).strip()))

    return all(is_numeric_like(gold) for gold in golds)


def _all_numeric_distractor_golds(golds) -> bool:
    def is_numeric_like(text: str) -> bool:
        return bool(re.fullmatch(r"[-+]?((\d+\.?\d*)|(\.\d+))", str(text).strip()))

    for gold_str in golds:
        gold_values = _parse_distractor_gold(gold_str)
        if not gold_values or not all(is_numeric_like(gold) for gold in gold_values):
            return False
    return True


def _score_with_judge(judge_model, judge_tok, task, predictions, golds, batch_size=8):
    """Score predictions using the Qwen 7B judge. Returns list of 0/1."""
    if task == "correct_answer":
        template = (
            "A student solved the following math problem and wrote this solution:\n"
            "{prediction}\n\n"
            "The correct final answer is: {gold}\n\n"
            "Did the student arrive at the correct final answer? "
            "Answer only 'yes' or 'no'."
        )
        prompts = [template.format(prediction=p, gold=g)
                   for p, g in zip(predictions, golds)]
    elif task == "next_subquestion":
        template = (
            "Are the following two math subquestions semantically equivalent?\n"
            "(They ask for exactly the same quantity, even if worded differently.)\n"
            "Answer only 'yes' or 'no'.\n\n"
            "Question 1: {gold}\n"
            "Question 2: {pred}"
        )
        prompts = [template.format(gold=g, pred=p)
                   for p, g in zip(predictions, golds)]
    elif task == "distractor":
        # Should not be called for distractor anymore (use _score_distractor_regex)
        template = (
            "A student was asked to produce an incorrect answer (a distractor) "
            "for a math problem and wrote the following reasoning:\n"
            "{prediction}\n\n"
            "The expected distractor answers are: {gold}\n\n"
            "Did the student arrive at any of the expected distractor answers? "
            "Answer only 'yes' or 'no'."
        )
        prompts = [template.format(prediction=p, gold=g)
                   for p, g in zip(predictions, golds)]
    else:
        raise ValueError(f"Unknown task: {task}")
    return _judge_yes_no(judge_model, judge_tok, prompts, batch_size)


def evaluate_checkpoints(
    args, val_prompts, val_golds, val_prompt_styles, quant_config, load_4bit,
):
    """Evaluate all epoch checkpoints on the val set using the judge model.

    For each checkpoint: load base model + LoRA adapter, generate predictions,
    free model. Then load judge once and score all checkpoints together.
    """
    ckpt_dirs = sorted(Path(args.out_dir).glob("checkpoint-*"))
    if not ckpt_dirs:
        print("No checkpoints found to evaluate.")
        return

    print(f"\n{'='*60}")
    print(f"Post-training evaluation: {len(ckpt_dirs)} checkpoints, "
          f"{len(val_prompts)} val prompts")
    print(f"{'='*60}")

    # Phase 1: generate predictions for each checkpoint
    all_predictions = {}  # ckpt_path -> list of predictions
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for ckpt_dir in ckpt_dirs:
        print(f"\nGenerating predictions for {ckpt_dir.name}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if not load_4bit else None,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(ckpt_dir))
        model = model.merge_and_unload()
        model.eval()

        preds = _generate_predictions(model, tokenizer, val_prompts, args.task)
        all_predictions[str(ckpt_dir)] = preds

        del model
        torch.cuda.empty_cache()

    # Phase 2: score all checkpoints
    # Distractor and EEDI correct-answer scoring handle their own routing.
    if args.task in {"distractor", "correct_answer"}:
        judge_model, judge_tok = None, None
    else:
        judge_model, judge_tok = _load_judge()

    best_acc, best_ckpt = -1.0, None
    for ckpt_path, preds in all_predictions.items():
        ckpt_name = Path(ckpt_path).name
        print(f"\nScoring {ckpt_name}...")
        if args.task == "distractor":
            scores = _score_distractor_regex(
                preds,
                val_golds,
                val_prompts,
                val_prompt_styles,
            )
        elif args.task == "correct_answer":
            scores = _score_correct_answer(
                preds,
                val_golds,
                val_prompts,
                val_prompt_styles,
            )
        else:
            scores = _score_with_judge(
                judge_model, judge_tok, args.task, preds, val_golds
            )
        accuracy = sum(scores) / len(scores) if scores else 0.0
        print(f"  {ckpt_name}: accuracy = {accuracy:.4f} ({sum(scores)}/{len(scores)})")

        with open(os.path.join(ckpt_path, "val_accuracy.json"), "w") as f:
            json.dump({
                "checkpoint": ckpt_name,
                "accuracy": accuracy,
                "n_correct": sum(scores),
                "n_samples": len(scores),
            }, f, indent=2)

        if accuracy > best_acc:
            best_acc = accuracy
            best_ckpt = ckpt_path

    if judge_model is not None:
        del judge_model
        torch.cuda.empty_cache()

    if best_ckpt:
        print(f"\nBest checkpoint: {best_ckpt} (accuracy={best_acc:.4f})")
        with open(os.path.join(args.out_dir, "best_checkpoint.json"), "w") as f:
            json.dump({"path": best_ckpt, "accuracy": best_acc}, f, indent=2)


# ── training ──────────────────────────────────────────────────────────────────

def train(args) -> None:
    df = pd.read_csv(args.train_csv)
    if "split" in df.columns:
        df = df[df["split"] == "train"].reset_index(drop=True)

    # correct_answer fallback: use reasoning_trace if target column is empty
    if args.task == "correct_answer" and "reasoning_trace" in df.columns:
        col = "target_question_reasoning_trace"
        if col in df.columns:
            mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            df.loc[mask, col] = df.loc[mask, "reasoning_trace"]
            if mask.any():
                print(f"  Fell back to 'reasoning_trace' for {mask.sum()} rows")

    # 80/20 train/val split (deterministic)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── quantization ──────────────────────────────────────────────────────
    load_4bit = args.load_in_4bit or _needs_4bit(args.model_name)
    quant_config = None
    if load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # ── model ─────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if not load_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # ── datasets ──────────────────────────────────────────────────────────
    train_dataset = build_hf_dataset(train_df, tokenizer, task=args.task)
    val_prompts, val_golds, val_prompt_styles = build_val_data(val_df, task=args.task)

    # ── training config ───────────────────────────────────────────────────
    training_kwargs = dict(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=args.epochs + 1,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    if _USE_SFT_CONFIG:
        import inspect as _inspect
        _sft_params = _inspect.signature(_SFTConfigCls.__init__).parameters
        _seq_len_kwarg = (
            "max_seq_length" if "max_seq_length" in _sft_params
            else "max_length" if "max_length" in _sft_params
            else None
        )
        if _seq_len_kwarg:
            training_kwargs[_seq_len_kwarg] = args.max_seq_len
        training_config = _SFTConfigCls(
            **training_kwargs,
            dataset_text_field="text",
        )
        trainer = SFTTrainer(
            model=model,
            args=training_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
    else:
        training_config = _SFTConfigCls(**training_kwargs)
        trainer = SFTTrainer(
            model=model,
            args=training_config,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_len,
            dataset_text_field="text",
        )

    # Auto-resume from latest checkpoint
    resume = any(
        d.name.startswith("checkpoint-")
        for d in Path(args.out_dir).iterdir()
        if d.is_dir()
    ) if Path(args.out_dir).exists() else False
    if resume:
        print(f"Resuming from latest checkpoint in {args.out_dir}")
    trainer.train(resume_from_checkpoint=resume)

    # Save final adapter + tokenizer
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved fine-tuned model → {args.out_dir}")

    # Free training model before evaluation
    del model, trainer
    torch.cuda.empty_cache()

    # Evaluate all checkpoints on val set with the judge model
    if not args.skip_eval:
        evaluate_checkpoints(
            args,
            val_prompts,
            val_golds,
            val_prompt_styles,
            quant_config,
            load_4bit,
        )
    else:
        print("Skipping post-training evaluation (--skip-eval).")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified fine-tuning with LoRA.")
    parser.add_argument("--model-name", required=True,
                        help="HuggingFace model id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--task", required=True,
                        choices=["correct_answer", "next_subquestion", "distractor"])
    parser.add_argument("--train-csv", required=True,
                        help="Path to training CSV")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to save checkpoints and final model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2560)
    parser.add_argument("--lora-r", type=int, default=64,
                        help="LoRA rank (default: 64)")
    parser.add_argument("--lora-alpha", type=int, default=128,
                        help="LoRA alpha (default: 128)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip post-training checkpoint evaluation")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Force 4-bit quantization (auto-enabled for 7B/8B)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
