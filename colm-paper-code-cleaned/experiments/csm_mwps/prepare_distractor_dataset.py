"""prepare_distractor_dataset.py

Generates two CSV files for the distractor fine-tuning experiment:
  1. correct_answer_distractor_pairs.csv — correct answer task (same synthetic tree problems)
  2. distractor_pairs.csv                — distractor generation task
                                           each row is one (question, distractor) pair;
                                           a question may appear multiple times if multiple
                                           misconception types are applicable.

Both CSVs share the same column conventions as the existing
correct_answer_pairs.csv / next_subquestion_pairs.csv so that
run_inference.py and finetune.py can consume them via --data-csv.

Run from: colm-paper-code-cleaned/experiments/csm_mwps/
  python prepare_distractor_dataset.py

Output (configurable via --out-dir):
  ../../reasoning-efficiency/experiments/proof_search/out/correct_answer_distractor_pairs.csv
  ../../reasoning-efficiency/experiments/proof_search/out/distractor_pairs.csv
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from datageneration import generate_ci_problems_and_distractors

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_DEFAULT_OUT = str("out")

TRAIN_RATIO = 0.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _misconception_label(misconceptions: dict) -> str:
    """Turn {node_id: RuleClassName, ...} into a readable string."""
    return ",".join(sorted(set(v for v in misconceptions.values())))


# Mapping from misconception class name to a human-readable operation word
_MISCONCEPTION_OPERATION = {
    "ContTransferContMisconceptionIncons": "transfer",
    "ContCompContMisconceptionIncons": "comparison",
}


def _annotate_twist_points(correct_rt: str, distractor_rt: str, misconception_type: str) -> str:
    """Insert 'Applying the twist to this <op> operation.' before each primary
    flip in a distractor reasoning trace.

    A primary flip is a sentence where the arithmetic operator (+/-) changed
    compared to the correct trace (same operands, different operator).
    Cascading differences (where only the numeric values change due to an
    earlier flip) are NOT annotated.
    """
    op_word = _MISCONCEPTION_OPERATION.get(misconception_type, "operation")
    marker = f"Applying the twist to this {op_word} operation."

    correct_sents = re.split(r'(?<=\.) ', correct_rt)
    dist_sents = re.split(r'(?<=\.) ', distractor_rt)

    annotated = []
    for j, ds in enumerate(dist_sents):
        cs = correct_sents[j] if j < len(correct_sents) else ""
        if cs != ds:
            # Check if this is a primary op flip
            c_ops = re.findall(r'(\d+) ([+\-*/]) (\d+) = (\d+)', cs)
            d_ops = re.findall(r'(\d+) ([+\-*/]) (\d+) = (\d+)', ds)
            if (c_ops and d_ops
                    and c_ops[0][0] == d_ops[0][0]
                    and c_ops[0][2] == d_ops[0][2]
                    and c_ops[0][1] != d_ops[0][1]):
                annotated.append(marker)
        annotated.append(ds)

    return " ".join(annotated)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate correct_answer + distractor CSV datasets")
    parser.add_argument("--n-problems", type=int, default=1000,
                        help="Number of MWP structures to generate (default: 1000)")
    parser.add_argument("--n-inst", type=int, default=5,
                        help="Number of numerical instantiations per MWP (default: 5)")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--out-dir", type=str, default=_DEFAULT_OUT,
                        help="Directory to write the two output CSVs")
    args = parser.parse_args()

    # ── generate raw dataset ────────────────────────────────────────────────
    print(f"Generating {args.n_problems} MWPs × {args.n_inst} instantiations "
          f"(seed={args.seed})…  (this may take several minutes)")

    dataset = generate_ci_problems_and_distractors(
        nr_problems=args.n_problems,
        nr_instantiations=args.n_inst,
        include_rt=True,          # include reasoning traces for correct + misconception answers
        include_cons_form=False,
        prob_misconcievable=1.0,  # select ALL eligible nodes so we get all 2^n-1 distractors
        seed=args.seed,
    )

    print(f"Generated {len(dataset)} MWPs.")

    # ── train / test split at problem level ─────────────────────────────────
    problem_ids = sorted(dataset.keys())
    rng = random.Random(args.seed)
    rng.shuffle(problem_ids)
    n_train = int(len(problem_ids) * TRAIN_RATIO)
    train_ids = set(problem_ids[:n_train])

    # ── flatten to rows ──────────────────────────────────────────────────────
    ca_rows   = []   # correct_answer_pairs_trees
    dist_rows = []   # distractor_pairs

    for mwp_id, mwp_data in tqdm(dataset.items(), desc="Building CSV rows"):
        split = "train" if mwp_id in train_ids else "test"

        for inst_id_str, inst in mwp_data["instantiations"].items():
            problem       = inst["problem"]
            correct_ans   = inst["correct_answer"]["answer"]
            correct_rt    = inst["correct_answer"].get("rt", "")

            # ── Correct-answer row ─────────────────────────────────────────
            ca_rows.append({
                "problem":                          problem,
                "correct_answer":                   correct_ans,
                "reasoning_trace":                  correct_rt,
                # target_answer          → used by run_inference.py for scoring
                "target_answer":                    correct_ans,
                # target_question_reasoning_trace → used by finetune.py as training label
                "target_question_reasoning_trace":  correct_rt,
                # prompt (bare question) → same convention as correct_answer_pairs.csv
                "prompt":                           problem,
                "split":                            split,
            })

            # ── Distractor rows: group by misconception TYPE ──────────────
            # Each misconception path may involve 1+ misconception types.
            # We only include a distractor in a type's row if ALL flipped
            # nodes in that path belong to that single type (no mixing).
            # E.g. if a path flips both a comparison and a transfer node,
            # it is excluded from both rows.
            # key: misconception_type -> list of (answer, rt)
            type_to_pairs = defaultdict(list)

            for misconception in inst.get("misconception_answers", []):
                if not misconception.get("plausible", False):
                    continue
                dist_answer = misconception["answer"]
                dist_rt     = misconception.get("rt", "")
                if not dist_rt:
                    continue

                # Only include if all flipped nodes share the same type
                misc_types = set(misconception.get("misconceptions", {}).values())
                if len(misc_types) == 1:
                    mtype = next(iter(misc_types))
                    type_to_pairs[mtype].append((dist_answer, dist_rt))

            prompt = f"Question: {problem}\nCorrect Answer: {correct_ans}"

            for mtype, pairs in type_to_pairs.items():
                # deduplicate by answer value, keep first reasoning trace
                seen = set()
                unique_answers = []
                unique_rts = []
                unique_explained_rts = []
                for ans, rt in pairs:
                    if ans not in seen:
                        seen.add(ans)
                        unique_answers.append(ans)
                        unique_rts.append(rt)
                        unique_explained_rts.append(
                            _annotate_twist_points(correct_rt, rt, mtype)
                        )

                # Skip misconception types producing >10 distractors
                if len(unique_answers) > 10:
                    continue

                dist_rows.append({
                    "problem":                           problem,
                    "correct_answer":                    correct_ans,
                    "reasoning_trace":                   correct_rt,
                    "distractor_answers":                json.dumps(unique_answers),
                    "distractor_reasoning_traces":       json.dumps(unique_rts),
                    "distractor_explained_traces":       json.dumps(unique_explained_rts),
                    "misconception_type":                mtype,
                    # target columns for run_inference / finetune
                    "target_distractor_answers":         json.dumps(unique_answers),
                    "target_distractor_reasoning_traces": json.dumps(unique_rts),
                    "target_distractor_explained_traces": json.dumps(unique_explained_rts),
                    "prompt":                            prompt,
                    "split":                             split,
                })

    ca_df   = pd.DataFrame(ca_rows)
    dist_df = pd.DataFrame(dist_rows)

    # Align: keep only correct-answer rows whose problem also has distractors
    if len(dist_df) > 0:
        dist_keys = set(zip(dist_df["problem"], dist_df["correct_answer"].astype(str)))
        ca_mask = ca_df.apply(
            lambda r: (r["problem"], str(r["correct_answer"])) in dist_keys, axis=1
        )
        n_dropped = (~ca_mask).sum()
        if n_dropped:
            print(f"  Dropped {n_dropped} correct_answer rows without valid distractors")
        ca_df = ca_df[ca_mask].reset_index(drop=True)

    # ── save ────────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    ca_path   = os.path.join(args.out_dir, "correct_answer_distractor_pairs.csv")
    dist_path = os.path.join(args.out_dir, "distractor_pairs.csv")

    ca_df.to_csv(ca_path,   index=False)
    dist_df.to_csv(dist_path, index=False)

    # ── summary ─────────────────────────────────────────────────────────────
    def _stats(df):
        tr = (df["split"] == "train").sum()
        te = (df["split"] == "test").sum()
        return f"{len(df)} rows  ({tr} train / {te} test)"

    print(f"\ncorrect_answer_distractor_pairs : {_stats(ca_df)}")
    print(f"  → {ca_path}")
    print(f"\ndistractor_pairs           : {_stats(dist_df)}")
    print(f"  → {dist_path}")


if __name__ == "__main__":
    main()
