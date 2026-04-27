import argparse
import gc
import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Qwen behavior on GSM8K: direct answers vs subquestion generation."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to test.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of GSM8K examples to inspect.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index in the GSM8K test split.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which GSM8K split to use.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="main",
        help="Dataset config for gsm8k.",
    )
    parser.add_argument(
        "--max_new_tokens_answer",
        type=int,
        default=256,
        help="Max new tokens for direct answering.",
    )
    parser.add_argument(
        "--max_new_tokens_subq",
        type=int,
        default=256,
        help="Max new tokens for subquestion generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Use 0.0 for greedy decoding.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for generation.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="qwen_gsm8k_inspection.jsonl",
        help="Where to save raw outputs.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model dtype.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> Optional[torch.dtype]:
    if dtype_str == "auto":
        return None
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def extract_gsm8k_final_answer(answer_text: str) -> str:
    """
    GSM8K answers usually end with '#### <number>'.
    """
    match = re.search(r"####\s*([^\n]+)", answer_text)
    if match:
        return match.group(1).strip()
    return answer_text.strip()


def normalize_number_string(text: str) -> Optional[str]:
    """
    Extract a simple final numeric-looking answer from model text.
    This is only for rough inspection, not rigorous evaluation.
    """
    if text is None:
        return None

    # Prefer explicit "Final answer: ..."
    patterns = [
        r"(?i)final answer\s*[:\-]?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"(?i)answer\s*[:\-]?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip())
        if match:
            return match.group(1).replace(",", "").strip()

    # Fallback: last number in the text
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if matches:
        return matches[-1].replace(",", "").strip()
    return None


def roughly_same_numeric_answer(pred: Optional[str], gold: Optional[str]) -> Optional[bool]:
    if pred is None or gold is None:
        return None

    def to_float(x: str) -> Optional[float]:
        try:
            return float(x.replace(",", "").replace("$", "").strip())
        except Exception:
            return None

    pred_f = to_float(pred)
    gold_f = to_float(gold)
    if pred_f is None or gold_f is None:
        return pred.strip() == gold.strip()
    return abs(pred_f - gold_f) < 1e-9


def build_answer_messages(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful math tutor. Solve the problem carefully. "
                "At the end, give a final line exactly in the format: Final answer: <number>"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]


def build_subquestion_messages(question: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a Socratic math tutor. Given a math word problem, generate a short sequence "
                "of guiding subquestions that help a student solve it step by step. "
                "Do NOT solve the problem. Do NOT reveal the final answer. "
                "Make the subquestions goal-oriented, concise, and in logical order."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{question}\n\n"
                "Generate 3 to 5 guiding subquestions only. "
                "Number them. Do not include explanations or the final answer."
            ),
        },
    ]


def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def load_model_and_tokenizer(model_name: str, dtype: Optional[torch.dtype]):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, tokenizer


def pretty_line(char: str = "=", width: int = 100) -> str:
    return char * width


def pretty_print_example_header(example_idx: int, question: str, gold_answer: str) -> None:
    print(pretty_line("="))
    print(f"EXAMPLE {example_idx}")
    print(pretty_line("-"))
    print("QUESTION:")
    print(question.strip())
    print(pretty_line("-"))
    print(f"GOLD FINAL ANSWER: {gold_answer}")
    print(pretty_line("="))


def pretty_print_model_block(
    model_name: str,
    answer_text: str,
    pred_answer: Optional[str],
    gold_answer: str,
    is_correct: Optional[bool],
    subq_text: str,
) -> None:
    print(pretty_line("#"))
    print(f"MODEL: {model_name}")
    print(pretty_line("-"))

    print("[DIRECT ANSWER OUTPUT]")
    print(answer_text.strip())
    print(pretty_line("."))

    print(f"Extracted predicted final answer: {pred_answer}")
    print(f"Gold final answer:              {gold_answer}")
    print(f"Rough numeric match:            {is_correct}")
    print(pretty_line("."))

    print("[GUIDING SUBQUESTIONS OUTPUT]")
    print(subq_text.strip())
    print(pretty_line("#"))
    print()


def cleanup_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    torch_dtype = get_torch_dtype(args.dtype)

    print("Loading GSM8K...")
    dataset = load_dataset("gsm8k", args.dataset_config, split=args.split)

    selected = dataset.select(range(args.start_index, args.start_index + args.num_examples))
    examples: List[Dict[str, str]] = []
    for ex in selected:
        examples.append(
            {
                "question": ex["question"].strip(),
                "gold_answer_full": ex["answer"].strip(),
                "gold_final_answer": extract_gsm8k_final_answer(ex["answer"]),
            }
        )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    all_records: List[Dict[str, Any]] = []

    for model_name in args.models:
        print(pretty_line("="))
        print(f"LOADING MODEL: {model_name}")
        print(pretty_line("="))

        model, tokenizer = load_model_and_tokenizer(model_name, torch_dtype)

        for idx, ex in enumerate(examples, start=1):
            question = ex["question"]
            gold_final_answer = ex["gold_final_answer"]

            answer_messages = build_answer_messages(question)
            subq_messages = build_subquestion_messages(question)

            answer_text = generate_one(
                model=model,
                tokenizer=tokenizer,
                messages=answer_messages,
                max_new_tokens=args.max_new_tokens_answer,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            subq_text = generate_one(
                model=model,
                tokenizer=tokenizer,
                messages=subq_messages,
                max_new_tokens=args.max_new_tokens_subq,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            pred_answer = normalize_number_string(answer_text)
            is_correct = roughly_same_numeric_answer(pred_answer, gold_final_answer)

            pretty_print_example_header(idx, question, gold_final_answer)
            pretty_print_model_block(
                model_name=model_name,
                answer_text=answer_text,
                pred_answer=pred_answer,
                gold_answer=gold_final_answer,
                is_correct=is_correct,
                subq_text=subq_text,
            )

            record = {
                "model_name": model_name,
                "example_index": idx,
                "question": question,
                "gold_answer_full": ex["gold_answer_full"],
                "gold_final_answer": gold_final_answer,
                "direct_answer_output": answer_text,
                "predicted_final_answer": pred_answer,
                "rough_numeric_match": is_correct,
                "subquestions_output": subq_text,
            }
            all_records.append(record)

        cleanup_model(model, tokenizer)

    with open(args.save_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(pretty_line("="))
    print(f"Saved raw outputs to: {args.save_path}")
    print(pretty_line("="))


if __name__ == "__main__":
    main()