#!/usr/bin/env python3
"""Evaluate OpenAI-generated EEDI reasoning traces with a Qwen 7B judge.

For each datapoint, this script checks whether the stored gold reasoning trace
arrives at the gold labelled final answer from the EEDI dataset.

Supported tasks:
  - correct_answer: uses target_question_reasoning_trace against correct_answer
  - distractor: uses target_distractor_explained_traces against
    target_distractor_answers

The script writes a per-trace CSV and prints a short summary.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_BATCH_SIZE = 8

JUDGE_CORE_INSTRUCTION = (
    "Your job is only to check whether the final value or expression stated in the reasoning trace is semantically equivalent "
    "to the provided gold label. Do not solve the problem yourself. Do not judge whether the intermediate reasoning is valid. "
    "First locate the final value or expression that the reasoning trace ends with. Then compare only that final value or expression against the gold label. "
    "Write your answer in exactly these three lines: 'Final value/expression in trace: ...', 'Gold label to compare: ...', and 'Conclusion: Equivalent -> Yes' or 'Conclusion: Not equivalent -> No'. "
    "If the intermediate reasoning is wrong but the final value or expression still matches the gold label, the verdict must be Yes. "
    "If the final value or expression does not match the gold label, the verdict must be No."
)

CORRECT_ANSWER_ONE_SHOT = (
    "Example:\n"
    "Question:\n"
    "Simplify the following, if possible: \\( \\frac{m^{2}+2 m-3}{m-3} \\)\n\n"
    "Reasoning trace:\n"
    "To simplify a rational expression, we look for common factors in the numerator and denominator.\n\n"
    "First factor the numerator:\n"
    "\\[\n"
    "m^2+2m-3\n"
    "\\]\n"
    "We need two numbers that multiply to \\(-3\\) and add to \\(2\\). Those numbers are \\(3\\) and \\(-1\\), so:\n"
    "\\[\n"
    "m^2+2m-3=(m+3)(m-1)\n"
    "\\]\n\n"
    "So the expression becomes:\n"
    "\\[\n"
    "\\frac{(m+3)(m-1)}{m-3}\n"
    "\\]\n\n"
    "The denominator is \\(m-3\\), and there is no matching factor of \\(m-3\\) in the numerator, so no cancellation is possible.\n\n"
    "Therefore, the expression does not simplify. The correct answer is: Does not simplify\n\n"
    "Gold final answer:\n"
    "Does not simplify\n\n"
    "Judge output:\n"
    "Final value/expression in trace: Does not simplify\n"
    "Gold label to compare: Does not simplify\n"
    "Conclusion: Equivalent -> Yes\n"
)

DISTRACTOR_ONE_SHOT = (
    "Example:\n"
    "Question:\n"
    "\\[\n"
    "3 \\times 2+4-5\n"
    "\\]\n"
    "Where do the brackets need to go to make the answer equal \\( 13 \\) ?\n\n"
    "Reasoning trace:\n"
    "Since addition comes before multiplication, I would do \\(2+4\\) first even without brackets:\n\n"
    "\\[\n"
    "3 \\times 2+4-5 = 3 \\times (2+4) - 5\n"
    "\\]\n\n"
    "Then\n\n"
    "\\[\n"
    "2+4=6,\\quad 3\\times 6=18,\\quad 18-5=13\n"
    "\\]\n\n"
    "So it already equals \\(13\\), so it does not need brackets.\n\n"
    "Gold target distractor answer:\n"
    "Does not need brackets\n\n"
    "Judge output:\n"
    "Final value/expression in trace: Does not need brackets\n"
    "Gold label to compare: Does not need brackets\n"
    "Conclusion: Equivalent -> Yes\n"
)


def _print_preview(records):
    for idx, record in enumerate(records, start=1):
        verdict = "Yes" if record["score"] else "No"
        print(f"\n=== Example {idx} ===")
        print(f"Task: {record['task']}")
        print(f"Question ID: {record.get('question_id')}")
        if "pair_index" in record and pd.notna(record["pair_index"]):
            print(f"Pair Index: {record['pair_index']}")
        if pd.notna(record.get("split")):
            print(f"Split: {record['split']}")
        print("Gold label:")
        print(record["gold_answer"])
        print("\nReasoning trace:")
        print(record["reasoning_trace"])
        print("\nQwen 7B judge output:")
        print(record["judge_output"])
        print(f"\nVerdict: {verdict}")


def _load_judge():
    print(f"Loading judge model: {JUDGE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        JUDGE_MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _judge_yes_no(judge_model, judge_tokenizer, prompts, batch_size):
    scores = []
    responses = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="LLM judge"):
        batch = prompts[start : start + batch_size]
        formatted = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch
        ]
        enc = judge_tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(judge_model.device)

        with torch.no_grad():
            out = judge_model.generate(
                **enc,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=judge_tokenizer.pad_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for seq in out:
            text = judge_tokenizer.decode(
                seq[input_len:],
                skip_special_tokens=True,
            ).strip()
            verdicts = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
            final_verdict = verdicts[-1].lower() if verdicts else "no"
            scores.append(1 if final_verdict == "yes" else 0)
            responses.append(text)
    return scores, responses


def _parse_list_field(raw_value: str):
    if pd.isna(raw_value):
        return []
    parsed = json.loads(raw_value)
    if isinstance(parsed, list):
        return [str(value) for value in parsed]
    return [str(parsed)]


def _normalize_answer(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^(answer:|final answer:|incorrect answer:|so the answer is)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_correct_answer_prompt(question: str, reasoning_trace: str, gold_answer: str) -> str:
    return (
        "You are judging whether the final value or expression stated in a reasoning trace matches the given gold final answer for a math problem. "
        f"{JUDGE_CORE_INSTRUCTION} "
        "Here is one example of the required behavior.\n\n"
        f"{CORRECT_ANSWER_ONE_SHOT}"
        "Now judge the next case. Keep the wording short and do not add any extra lines before or after those three lines.\n\n"
        f"Question:\n{question}\n\n"
        f"Reasoning trace:\n{reasoning_trace}\n\n"
        f"Gold final answer:\n{_normalize_answer(gold_answer)}\n"
        "Your judge:\n"
    )


def _build_distractor_prompt(question: str, reasoning_trace: str, gold_answer: str) -> str:
    return (
        "You are judging whether the final value or expression stated in a student-style incorrect reasoning trace matches the given gold target distractor answer for a math problem. "
        f"{JUDGE_CORE_INSTRUCTION} "
        "Here is one example of the required behavior.\n\n"
        f"{DISTRACTOR_ONE_SHOT}"
        "Now judge the next case. Keep the wording short and do not add any extra lines before or after those three lines.\n\n"
        f"Question:\n{question}\n\n"
        f"Reasoning trace:\n{reasoning_trace}\n\n"
        f"Gold target distractor answer:\n{_normalize_answer(gold_answer)}\n"
        "Your judge:\n"
    )


def _build_examples(df: pd.DataFrame, task: str):
    examples = []
    if task == "correct_answer":
        for row_idx, row in df.iterrows():
            trace = str(row["target_question_reasoning_trace"]).strip()
            if not trace:
                continue
            examples.append(
                {
                    "row_idx": row_idx,
                    "trace_idx": 0,
                    "question_id": row.get("question_id"),
                    "split": row.get("split"),
                    "question": str(row["question"]),
                    "gold_answer": str(row["correct_answer"]),
                    "reasoning_trace": trace,
                    "prompt": _build_correct_answer_prompt(
                        str(row["question"]),
                        trace,
                        str(row["correct_answer"]),
                    ),
                }
            )
        return examples

    if task == "distractor":
        for row_idx, row in df.iterrows():
            traces = _parse_list_field(row["target_distractor_explained_traces"])
            gold_answers = _parse_list_field(row["target_distractor_answers"])
            gold_answer = gold_answers[0] if gold_answers else ""
            for trace_idx, trace in enumerate(traces):
                trace = str(trace).strip()
                if not trace:
                    continue
                examples.append(
                    {
                        "row_idx": row_idx,
                        "trace_idx": trace_idx,
                        "question_id": row.get("question_id"),
                        "pair_index": row.get("pair_index"),
                        "split": row.get("split"),
                        "question": str(row["question"]),
                        "gold_answer": gold_answer,
                        "reasoning_trace": trace,
                        "prompt": _build_distractor_prompt(
                            str(row["question"]),
                            trace,
                            gold_answer,
                        ),
                    }
                )
        return examples

    raise ValueError(f"Unsupported task: {task}")


def _default_task_from_csv(csv_path: Path) -> str:
    name = csv_path.name.lower()
    if "correct_answer" in name:
        return "correct_answer"
    if "distractor" in name:
        return "distractor"
    raise ValueError("Could not infer task from CSV filename. Pass --task explicitly.")


def main():
    parser = argparse.ArgumentParser(
        description="Judge whether stored EEDI reasoning traces reach the gold final answer.",
    )
    parser.add_argument(
        "--data-csv",
        required=True,
        help="Path to correct_answer_pairs_eedi*.csv or distractor_pairs_eedi*.csv",
    )
    parser.add_argument(
        "--task",
        choices=["correct_answer", "distractor"],
        default=None,
        help="Optional explicit task. If omitted, inferred from CSV filename.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split filter, e.g. train / validation / test.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap before trace expansion.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--preview-examples",
        type=int,
        default=0,
        help="If > 0, run only the first N expanded trace examples and print full judge details to the terminal.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Path for per-trace evaluation output CSV.",
    )
    args = parser.parse_args()

    csv_path = Path(args.data_csv)
    out_path = Path(args.out_csv) if args.out_csv else None
    task = args.task or _default_task_from_csv(csv_path)

    df = pd.read_csv(csv_path)
    if args.split is not None:
        df = df[df["split"] == args.split].copy()
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()
    df = df.reset_index(drop=True)

    examples = _build_examples(df, task)
    if args.preview_examples > 0:
        examples = examples[: args.preview_examples]
    print(f"Task: {task} | Rows: {len(df)} | Trace examples: {len(examples)}")
    if not examples:
        raise ValueError("No reasoning traces found to evaluate.")

    judge_model, judge_tokenizer = _load_judge()
    scores, judge_outputs = _judge_yes_no(
        judge_model,
        judge_tokenizer,
        [example["prompt"] for example in examples],
        args.batch_size,
    )

    records = []
    for example, score, judge_output in zip(examples, scores, judge_outputs):
        record = {
            "task": task,
            "row_idx": example["row_idx"],
            "trace_idx": example["trace_idx"],
            "question_id": example.get("question_id"),
            "split": example.get("split"),
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "reasoning_trace": example["reasoning_trace"],
            "score": score,
            "verdict": "Yes" if score else "No",
            "judge_output": judge_output,
        }
        if "pair_index" in example:
            record["pair_index"] = example["pair_index"]
        records.append(record)

    out_df = pd.DataFrame(records)
    if args.preview_examples > 0:
        _print_preview(records)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved per-trace results to {out_path}")

    accuracy = out_df["score"].mean()
    print(f"Accuracy: {accuracy:.1%} ({int(out_df['score'].sum())}/{len(out_df)})")
    if "split" in out_df.columns:
        split_scores = out_df.groupby("split")["score"].agg(["mean", "count", "sum"]).reset_index()
        print("By split:")
        for _, row in split_scores.iterrows():
            print(f"  {row['split']}: {row['mean']:.1%} ({int(row['sum'])}/{int(row['count'])})")


if __name__ == "__main__":
    main()