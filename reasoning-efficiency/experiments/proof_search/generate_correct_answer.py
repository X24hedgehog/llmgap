import json
from pathlib import Path
from typing import Any, Dict, List

import click
import pandas as pd
from tqdm import tqdm

from generate_next_subquestion import _base_columns, _build_examples_for_row, _parse_jsonish, _serialize


DEFAULT_INPUT_PATH = Path("out/full_data.csv")
DEFAULT_OUTPUT_PATH = Path("out/correct_answer_pairs.csv")


def _parse_answer_mapping(row: Dict[str, Any]) -> Dict[str, Any]:
    parsed = _parse_jsonish(row.get("subquestion_answers_by_node_id"))
    if not isinstance(parsed, dict):
        return {}
    return {str(key): value for key, value in parsed.items()}


def _parse_subproblem_mapping(row: Dict[str, Any]) -> Dict[str, str]:
    parsed = _parse_jsonish(row.get("subproblems_by_node_id"))
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(value) for key, value in parsed.items()}


def build_correct_answer_pairs(input_df: pd.DataFrame) -> pd.DataFrame:
    pair_rows: List[Dict[str, Any]] = []

    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Rows"):
        row_dict = row.to_dict()
        answer_mapping = _parse_answer_mapping(row_dict)
        subproblem_mapping = _parse_subproblem_mapping(row_dict)
        if not answer_mapping:
            raise ValueError(
                "Missing subquestion_answers_by_node_id in input data. Regenerate out/data.csv with the updated generate_csv.py first."
            )
        if not subproblem_mapping:
            raise ValueError(
                "Missing subproblems_by_node_id in input data. Regenerate out/full_data.csv with the updated generate_csv.py first."
            )

        examples = _build_examples_for_row(row_dict)
        for pair_index, example in enumerate(examples):
            target_node_id = example["target_node_id"]
            target_answer = answer_mapping.get(str(target_node_id))
            target_problem = subproblem_mapping.get(str(target_node_id))
            if target_answer is None:
                raise ValueError(
                    f"Missing target answer for node_id={target_node_id} in example_idx={row_dict.get('example_idx')}"
                )
            if target_problem is None:
                raise ValueError(
                    f"Missing target subproblem for node_id={target_node_id} in example_idx={row_dict.get('example_idx')}"
                )

            pair_rows.append(
                {
                    **_base_columns(row_dict),
                    "pair_index": pair_index,
                    "solved_node_ids": _serialize(example["solved_node_ids"]),
                    "solved_steps": _serialize(example["solved_steps"]),
                    "target_node_id": target_node_id,
                    "target_question": example["next_subquestion"],
                    "target_problem": target_problem,
                    "target_answer": target_answer,
                    "prompt": target_problem,
                }
            )

    return pd.DataFrame(pair_rows)


@click.command()
@click.option("--input-path", default=str(DEFAULT_INPUT_PATH), show_default=True, help="Flattened reasoning-efficiency CSV with subquestion answers")
@click.option("--out-path", default=str(DEFAULT_OUTPUT_PATH), show_default=True, help="Output path for the aligned correct-answer CSV")
@click.option("--limit", default=None, type=int, help="Optional row limit for debugging")
def cli(input_path: str, out_path: str, limit: int):
    input_df = pd.read_csv(input_path)
    if limit is not None:
        input_df = input_df.head(limit)

    pair_df = build_correct_answer_pairs(input_df)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(out_file, index=False)


if __name__ == "__main__":
    cli()
