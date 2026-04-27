#!/usr/bin/env python3
"""Manually test the local Qwen 7B judge on hand-crafted subquestion pairs.

Edit the PAIRS list below, then run:
    python test_similarity_judge.py

Each entry is (gold, prediction). The judge decides if they are semantically
equivalent (same quantity asked, even if worded differently).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ── Edit these pairs ──────────────────────────────────────────────────────────
PAIRS = [
    (
        "How many violins does Hannah have after giving Matthias 11 violins?",
        "What is the number of violins Hannah has after she gives 11 violins to Matthias?",
    ),
    (
        "How many chairs does Rockie have after giving Karlis 5 chairs?",
        "How many chairs does Rockie have?",
    ),
    (
        "How many apples does Alice have in total?",
        "What is the total number of apples Alice owns?",
    ),
    # Add more pairs here:
    # ("gold question", "predicted question"),
]
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT = (
    "Are the following two math subquestions semantically equivalent?\n"
    "(They ask for exactly the same quantity, even if worded differently.)\n"
    "Answer only 'yes' or 'no'.\n\n"
    "Question 1: {gold}\n"
    "Question 2: {pred}"
)


def load_judge():
    print(f"Loading judge: {JUDGE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME, trust_remote_code=True)
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


def judge_pair(pred: str, gold: str, model, tokenizer) -> tuple[str, int]:
    """Return (raw_output, score) where score=1 means equivalent."""
    prompt = JUDGE_PROMPT.format(gold=gold, pred=pred)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = enc["input_ids"].shape[1]
    raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip().lower()
    return raw, (1 if raw.startswith("yes") else 0)


def main():
    model, tokenizer = load_judge()

    print(f"\nJudge model : {JUDGE_MODEL_NAME}")
    print(f"Testing {len(PAIRS)} pair(s)")
    print("=" * 72)

    scores = []
    for i, (gold, pred) in enumerate(PAIRS):
        raw, score = judge_pair(pred, gold, model, tokenizer)
        scores.append(score)
        verdict = "✓ YES (same)" if score == 1 else "✗ NO  (different)"
        print(f"\n[{i+1:>3}] {verdict}  (judge said: '{raw}')")
        print(f"  Gold : {gold}")
        print(f"  Pred : {pred}")

    print("\n" + "=" * 72)
    print(f"Score: {sum(scores)}/{len(scores)} pairs judged equivalent")


if __name__ == "__main__":
    main()
