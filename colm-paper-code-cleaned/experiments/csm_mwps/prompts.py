"""Shared prompt templates for correct-answer and distractor tasks.

Import from here in both run_inference.py and finetune.py to keep prompts in sync.
"""

CORRECT_ANSWER_INSTRUCTION = "\n\nSolve this problem step by step."

# Misconception-specific distractor prompts (0-shot with step-by-step instructions)
DISTRACTOR_PROMPT_BY_TYPE = {
    "ContCompContMisconceptionIncons": (
        "You are an expert math educator generating distractors based on the Comparison Inconsistency misconception.\n\n"
        "When a problem says \"A has N more than B\", the correct way to find B is B = A - N.\n"
        "A student with this misconception FLIPS the operation: they compute B = A + N instead.\n"
        "Similarly, \"fewer than\" → they add instead of subtract, "
        "\"times as many\" → they divide instead of multiply, etc.\n\n"
        "Your job is to simulate these mistakes and generate all possible distractors of the following math question.\n\n"
        "Question: {problem}\n"
        "Think step by step:\n\n"
        "Step 1 — Locate every comparison statement in the problem (phrases like "
        "\"more than\", \"fewer than\", \"less than\", \"times as many\"). "
        "Label them (A), (B), (C), … and for each one, show the correct operation "
        "and the flipped (misconceived) operation.\n\n"
        "Step 2 — Enumerate every non-empty subset of those comparison steps. "
        "For each subset, recompute the final answer by flipping ONLY the operations "
        "in that subset while keeping everything else correct. "
        "State which steps you are flipping before each computation.\n\n"
        "Step 3 — Collect all the distractor answers."
    ),
    "ContTransferContMisconceptionIncons": (
        "You are an expert math educator generating distractors based on the Transfer Inconsistency misconception.\n\n"
        "MISCONCEPTION — Transfer Inconsistency:\n"
        "When a problem says \"A gives B N items\", the correct effect on A is: A loses N (subtract).\n"
        "A student with this misconception FLIPS the operation: they add instead of subtract "
        "(or subtract instead of add), confusing the direction of the transfer.\n"
        "Similarly, \"sells to\" → they add instead of subtract for the seller, "
        "\"buys from\" → they subtract instead of add for the buyer, etc.\n\n"
        "Your job is to simulate these mistakes and generate all possible distractors of the following math question.\n\n"
        "Question: {problem}\n"
        "Think step by step:\n\n"
        "Step 1 — Locate every transfer statement in the problem (phrases like "
        "\"gives\", \"receives\", \"sells to\", \"buys from\"). "
        "Label them (A), (B), (C), … and for each one, show the correct operation "
        "and the flipped (misconceived) operation.\n\n"
        "Step 2 — Enumerate every non-empty subset of those transfer steps. "
        "For each subset, recompute the final answer by flipping ONLY the operations "
        "in that subset while keeping everything else correct. "
        "State which steps you are flipping before each computation.\n\n"
        "Step 3 — Collect all the distractor answers."
    ),
}
