#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_ORM_MODEL = "RLHFlow/Llama3.1-8B-ORM-Mistral-Data"


# =============================================================================
# Manual fake generations to test ORM behavior
# =============================================================================
# For each dataset row_idx, we include:
#   - exact/paraphrased/equivalent candidates: should get high ORM score
#   - too-far-ahead candidates: often plausible but not the labelled next subquestion
#   - irrelevant/wrong candidates: should get low ORM score
#
# You can manually edit this dictionary to stress-test the ORM.
# =============================================================================

MANUAL_CANDIDATES: dict[int, list[dict[str, str]]] = {
    0: [
        {
            "label": "exact_gold",
            "candidate": "How many clips did Natalia sell in May?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "How many clips did Natalia sell during May?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "Since May sales were half of April sales, how many clips were sold in May?",
        },
        {
            "label": "too_far_ahead_final_question",
            "candidate": "How many clips did Natalia sell altogether in April and May?",
        },
        {
            "label": "wrong_irrelevant",
            "candidate": "How much money did Natalia earn from selling clips?",
        },
    ],
    1: [
        {
            "label": "exact_gold",
            "candidate": "How much does Weng earn per minute for babysitting?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "What is Weng's babysitting rate per minute?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "If Weng earns $12 per hour, how much does she earn in one minute?",
        },
        {
            "label": "too_far_ahead_later_step",
            "candidate": "How much did Weng earn from babysitting for 50 minutes?",
        },
        {
            "label": "wrong_irrelevant",
            "candidate": "How many hours did Weng babysit yesterday?",
        },
    ],
    2: [
        {
            "label": "exact_gold",
            "candidate": "How much money did Betty's grandparents give her?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "How much did Betty receive from her grandparents?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "Since Betty's grandparents gave twice as much as her parents, how much money did they give her?",
        },
        {
            "label": "previous_step_already_known",
            "candidate": "How much money does Betty initially have?",
        },
        {
            "label": "too_far_ahead_final_question",
            "candidate": "How much more money does Betty need to buy the wallet?",
        },
        {
            "label": "wrong_irrelevant",
            "candidate": "How much does Betty spend on food?",
        },
    ],
    3: [
        {
            "label": "exact_gold",
            "candidate": "How many pages has Julie read so far?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "How many pages has Julie read in total by the end of today?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "After reading yesterday and today, how many pages has Julie finished?",
        },
        {
            "label": "previous_step_already_known",
            "candidate": "How many pages did Julie read today?",
        },
        {
            "label": "too_far_ahead_later_step",
            "candidate": "How many pages are left for Julie to read?",
        },
        {
            "label": "too_far_ahead_final_question",
            "candidate": "How many pages should Julie read tomorrow?",
        },
    ],
    4: [
        {
            "label": "exact_gold",
            "candidate": "How many pages does James write to both friends in a week?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "What is the total number of pages James writes each week to both friends?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "After accounting for both friends, how many pages does James write per week?",
        },
        {
            "label": "previous_step_already_known",
            "candidate": "How many pages does James write to each friend in a week?",
        },
        {
            "label": "too_far_ahead_final_question",
            "candidate": "How many pages does James write in a year?",
        },
        {
            "label": "wrong_irrelevant",
            "candidate": "How many friends does James have at school?",
        },
    ],
    5: [
        {
            "label": "exact_gold",
            "candidate": "How many purple flowers are there in total?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "What is the total number of purple flowers in Mark's garden?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "Given the extra purple flowers, how many purple flowers does Mark have altogether?",
        },
        {
            "label": "previous_step_already_known",
            "candidate": "How many more purple flowers are there compared to yellow flowers?",
        },
        {
            "label": "too_far_ahead_later_step",
            "candidate": "How many flowers are there in total between yellow and purple flowers?",
        },
        {
            "label": "too_far_ahead_final_question",
            "candidate": "How many flowers does Mark have in his garden in total?",
        },
    ],
    6: [
        {
            "label": "exact_gold",
            "candidate": "How many slices of pizza does Albert eat from the two small pizzas?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "How many total slices are in Albert's two small pizzas?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "Since each small pizza has 8 slices, how many slices come from the two small pizzas?",
        },
        {
            "label": "previous_step_already_known",
            "candidate": "How many slices of pizza does Albert eat from the two large pizzas?",
        },
        {
            "label": "too_far_ahead_final_question",
            "candidate": "How many total slices of pizza does Albert eat in one day?",
        },
        {
            "label": "wrong_irrelevant",
            "candidate": "How much money did Albert pay for the pizzas?",
        },
    ],
    7: [
        {
            "label": "exact_gold",
            "candidate": "What was the weight of the box after adding another 2 pounds of jelly beans?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "After the brownies and then 2 more pounds of jelly beans are added, what is the box's weight?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "What is the package weight after increasing it to 6 pounds and adding 2 more pounds of jelly beans?",
        },
        {
            "label": "previous_step_already_known",
            "candidate": "What was the weight of the box after adding enough brownies to triple the initial 2 pounds of jelly beans?",
        },
        {
            "label": "too_far_ahead_final_question",
            "candidate": "What was the final weight of the box after adding enough gummy worms?",
        },
        {
            "label": "wrong_irrelevant",
            "candidate": "How many jelly beans did Ken put in the box?",
        },
    ],
    8: [
        {
            "label": "exact_gold_or_expected",
            "candidate": "How much did Alexis spend on the shoes and other items combined?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "How much money did Alexis spend in total before subtracting what she had left?",
        },
        {
            "label": "equivalent_paraphrase",
            "candidate": "Given that Alexis had $16 left from her $200 budget, how much did she spend altogether?",
        },
        {
            "label": "alternative_useful_step",
            "candidate": "How much did Alexis spend on all the listed clothing items except the shoes?",
        },
        {
            "label": "too_far_ahead_final_question",
            "candidate": "How much did Alexis pay for the shoes?",
        },
        {
            "label": "wrong_irrelevant",
            "candidate": "What color were Alexis's business clothes?",
        },
    ],
}


def make_quant_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


class RLHFlowMathOutcomeRewardModel:
    """
    Stable RLHFlow ORM decoding.

    The ORM is a causal LM. We turn it into a binary judge by asking it
    to output either "+" or "-".

    For each judge prompt, we build two conversations:

    Real:
        user: <judge prompt>
        assistant: +

    Mask:
        user: <judge prompt>
        assistant: ки

    The dummy token "ки" is only used to locate the assistant-label position.
    Then we read P("+") from the real conversation logits one position earlier.
    """

    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool,
    ) -> None:
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        quant_config = make_quant_config(load_in_4bit)

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
                f"Expected single-token +/- labels, got plus={plus_tag_id}, minus={minus_tag_id}"
            )

        self.candidate_tokens = [plus_tag_id[0], minus_tag_id[0]]

        self.special_tok_id = int(
            self.tokenizer("ки", return_tensors="pt").input_ids[0, 1]
        )

        print(f"Loaded ORM: {model_name}", flush=True)
        print(f"ORM candidate token ids [+,-]: {self.candidate_tokens}", flush=True)
        print(f"ORM dummy token id: {self.special_tok_id}", flush=True)

    def _build_pair(
        self,
        judge_prompt: str,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        conversation = [
            {"role": "user", "content": judge_prompt},
            {"role": "assistant", "content": "+"},
        ]

        conversation_mask = [
            {"role": "user", "content": judge_prompt},
            {"role": "assistant", "content": "ки"},
        ]

        return conversation, conversation_mask

    @torch.no_grad()
    def score_judge_prompts(
        self,
        judge_prompts: list[str],
        batch_size: int,
    ) -> list[float]:
        pairs = [self._build_pair(prompt) for prompt in judge_prompts]
        scores: list[float] = []

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]

            conversations = [pair[0] for pair in batch_pairs]
            conversations_mask = [pair[1] for pair in batch_pairs]

            input_ids = self.tokenizer.apply_chat_template(
                conversations,
                padding=True,
                return_tensors="pt",
                return_dict=False,
            ).to(self.model.device)

            input_ids_mask = self.tokenizer.apply_chat_template(
                conversations_mask,
                padding=True,
                return_tensors="pt",
                return_dict=False,
            ).to(self.model.device)

            if input_ids.shape != input_ids_mask.shape:
                raise RuntimeError(
                    f"ORM shape mismatch: real={tuple(input_ids.shape)}, "
                    f"mask={tuple(input_ids_mask.shape)}"
                )

            logits = self.model(input_ids).logits[:, :, self.candidate_tokens]
            probs_plus = logits.softmax(dim=-1)[:, :, 0]

            for row_idx in range(input_ids.shape[0]):
                # Causal shift:
                # token at position t is predicted by logits at position t-1.
                label_mask = input_ids_mask[row_idx, 1:] == self.special_tok_id
                plus_probs = (
                    probs_plus[row_idx, :-1][label_mask]
                    .float()
                    .detach()
                    .cpu()
                    .tolist()
                )

                if len(plus_probs) != 1:
                    raise RuntimeError(
                        f"Expected exactly one ORM label position, got {len(plus_probs)} "
                        f"for sample index {start + row_idx}"
                    )

                scores.append(float(plus_probs[0]))

        return scores


def build_judge_prompt(
    question: str,
    gold_subquestion: str,
    candidate_subquestion: str,
    reasoning_trace: str = "",
    tree: str = "",
    include_trace: bool = False,
    include_tree: bool = False,
) -> str:
    context_parts = ""

    if include_trace:
        context_parts += f"""

Full solution trace, for context only:
{reasoning_trace}"""

    if include_tree:
        context_parts += f"""

Decomposition tree, for context only:
{tree}"""

    return f"""You are judging a generated next subquestion for a math word problem.

Original math problem:
{question}

Reference/golden next subquestion:
{gold_subquestion}

Candidate generated next subquestion:
{candidate_subquestion}
{context_parts}

Judge whether the candidate generated subquestion is semantically equivalent to, or at least as useful as, the reference/golden next subquestion.

A good candidate should:
- ask the same mathematical intermediate question, or an equivalent one
- be useful as the next step toward solving the original problem
- not ask a previous step that is already known
- not jump directly to the final answer if the reference asks an intermediate step
- not contain irrelevant explanation

Output "+" if the candidate is good.
Output "-" if the candidate is bad."""


def load_examples(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.data_csv)

    required = {
        args.question_col,
        args.subquestion_col,
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available columns: {list(df.columns)}"
        )

    if args.split_col and args.split_col in df.columns:
        df = df[
            df[args.split_col].astype(str).str.lower()
            == args.split.lower()
        ].reset_index(drop=True)

    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].reset_index(drop=True)

    return df


def build_candidates_for_row(
    row_idx: int,
    gold_subquestion: str,
) -> list[dict[str, str]]:
    if row_idx in MANUAL_CANDIDATES:
        candidates = MANUAL_CANDIDATES[row_idx]
    else:
        candidates = [
            {
                "label": "exact_gold",
                "candidate": gold_subquestion,
            },
            {
                "label": "too_far_ahead_final_question",
                "candidate": "What is the final answer to the whole problem?",
            },
            {
                "label": "wrong_irrelevant",
                "candidate": "How many apples are there?",
            },
        ]

    # Ensure exact gold is always present.
    candidate_texts = [item["candidate"].strip() for item in candidates]
    if gold_subquestion.strip() not in candidate_texts:
        candidates = [
            {
                "label": "exact_gold_auto_added",
                "candidate": gold_subquestion,
            }
        ] + candidates

    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test RLHFlow ORM reward quality on next-subquestion candidates."
    )

    parser.add_argument(
        "--data-csv",
        default="/cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search/out/next_subquestion_pairs_gsm8k.csv",
    )

    parser.add_argument("--orm-model-name", default=DEFAULT_ORM_MODEL)
    parser.add_argument("--orm-load-in-4bit", action="store_true")
    parser.add_argument("--orm-batch-size", type=int, default=1)

    parser.add_argument("--question-col", default="question")
    parser.add_argument("--subquestion-col", default="next_subquestion")
    parser.add_argument("--reasoning-trace-col", default="reasoning_trace")
    parser.add_argument("--tree-col", default="tree")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--split", default="train")

    parser.add_argument("--max-rows", type=int, default=9)

    parser.add_argument(
        "--include-trace",
        action="store_true",
        help="Include full reasoning_trace in ORM judge prompt.",
    )
    parser.add_argument(
        "--include-tree",
        action="store_true",
        help="Include decomposition tree in ORM judge prompt.",
    )

    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional path to save scores as CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_examples(args)

    print(f"Loaded examples: {len(df)}", flush=True)
    print(f"Dataset: {args.data_csv}", flush=True)

    orm = RLHFlowMathOutcomeRewardModel(
        model_name=args.orm_model_name,
        load_in_4bit=args.orm_load_in_4bit,
    )

    all_judge_prompts: list[str] = []
    metadata: list[dict[str, Any]] = []

    for row_idx, row in df.iterrows():
        question = str(row[args.question_col])
        gold = str(row[args.subquestion_col])

        reasoning_trace = (
            str(row[args.reasoning_trace_col])
            if args.reasoning_trace_col in row.index
            else ""
        )
        tree = str(row[args.tree_col]) if args.tree_col in row.index else ""

        candidate_items = build_candidates_for_row(
            row_idx=row_idx,
            gold_subquestion=gold,
        )

        for candidate_idx, item in enumerate(candidate_items):
            candidate = item["candidate"]
            label = item["label"]

            judge_prompt = build_judge_prompt(
                question=question,
                gold_subquestion=gold,
                candidate_subquestion=candidate,
                reasoning_trace=reasoning_trace,
                tree=tree,
                include_trace=args.include_trace,
                include_tree=args.include_tree,
            )

            all_judge_prompts.append(judge_prompt)

            metadata.append(
                {
                    "row_idx": row_idx,
                    "candidate_idx": candidate_idx,
                    "candidate_label": label,
                    "question": question,
                    "gold_subquestion": gold,
                    "candidate_subquestion": candidate,
                    "is_exact_gold": candidate.strip() == gold.strip(),
                }
            )

    scores = orm.score_judge_prompts(
        all_judge_prompts,
        batch_size=args.orm_batch_size,
    )

    rows = []
    for meta, score in zip(metadata, scores):
        row = dict(meta)
        row["orm_score_p_plus"] = score
        rows.append(row)

    result_df = pd.DataFrame(rows)

    print("\n" + "=" * 120, flush=True)
    print("ORM SCORING RESULTS", flush=True)
    print("=" * 120, flush=True)

    for row_idx, group_df in result_df.groupby("row_idx"):
        first = group_df.iloc[0]

        print("\n" + "-" * 120, flush=True)
        print(f"ROW {row_idx}", flush=True)
        print("QUESTION:", flush=True)
        print(first["question"], flush=True)
        print("\nGOLD NEXT SUBQUESTION:", flush=True)
        print(first["gold_subquestion"], flush=True)
        print("\nCANDIDATE SCORES, SORTED HIGH TO LOW:", flush=True)

        group_df = group_df.sort_values(
            by="orm_score_p_plus",
            ascending=False,
        )

        for _, r in group_df.iterrows():
            exact = " EXACT_GOLD" if bool(r["is_exact_gold"]) else ""
            print(
                f"  score={r['orm_score_p_plus']:.6f} | "
                f"{r['candidate_label']}{exact} | "
                f"{r['candidate_subquestion']}",
                flush=True,
            )

    print("\n" + "=" * 120, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 120, flush=True)

    exact_scores = result_df[result_df["is_exact_gold"]]["orm_score_p_plus"]
    non_exact_scores = result_df[~result_df["is_exact_gold"]]["orm_score_p_plus"]

    if len(exact_scores) > 0:
        print(f"Mean exact-gold score: {exact_scores.mean():.6f}", flush=True)

    if len(non_exact_scores) > 0:
        print(f"Mean non-exact candidate score: {non_exact_scores.mean():.6f}", flush=True)

    label_summary = (
        result_df.groupby("candidate_label")["orm_score_p_plus"]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    print("\nMean score by candidate label:", flush=True)
    print(label_summary.to_string(index=False), flush=True)

    if args.out_csv:
        result_df.to_csv(args.out_csv, index=False)
        print(f"\nSaved scores to: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()