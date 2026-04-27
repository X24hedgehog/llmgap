import ast
import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Set, Tuple

import click
import pandas as pd
from tqdm import tqdm


DEFAULT_INPUT_PATH = Path("out/full_data.csv")
DEFAULT_COMPACT_OUTPUT_PATH = Path("out/next_subquestion_compact.csv")
DEFAULT_PAIRS_OUTPUT_PATH = Path("out/next_subquestion_pairs.csv")

COMPACT_MAPPING_COLUMN = "solved_steps_to_next_subquestion"


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return text
    return value


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _normalize_question_text(question: str) -> str:
    return " ".join(question.strip().split())


def _collect_tree_metadata(subquestion_tree: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[int, Set[int]], Dict[int, int]]:
    question_by_node_id: Dict[int, str] = {}
    child_subquestions_by_node_id: Dict[int, Set[int]] = {}
    rank_by_node_id: Dict[int, int] = {}

    def _visit(node: Dict[str, Any]) -> int:
        node_id = int(node["node_id"])
        question = node.get("question")
        subquestion_children: Set[int] = set()
        child_ranks: List[int] = []

        for child in node.get("children", []):
            child_rank = _visit(child)
            child_ranks.append(child_rank)
            child_question = child.get("question")
            if child_question:
                subquestion_children.add(int(child["node_id"]))

        if question:
            question_by_node_id[node_id] = str(question)
            child_subquestions_by_node_id[node_id] = subquestion_children
            rank_by_node_id[node_id] = 0 if not child_ranks else max(child_ranks) + 1

        return 0 if not child_ranks else max(child_ranks) + (1 if question else 0)

    _visit(subquestion_tree)
    return question_by_node_id, child_subquestions_by_node_id, rank_by_node_id


def _sorted_node_ids(node_ids: Set[int], rank_by_node_id: Dict[int, int]) -> List[int]:
    return sorted(node_ids, key=lambda node_id: (rank_by_node_id[node_id], node_id))


def _available_nodes(all_node_ids: Set[int], child_subquestions_by_node_id: Dict[int, Set[int]],
                     solved_node_ids: Set[int], rank_by_node_id: Dict[int, int]) -> List[int]:
    available = [
        node_id
        for node_id in all_node_ids
        if node_id not in solved_node_ids and child_subquestions_by_node_id[node_id].issubset(solved_node_ids)
    ]
    return _sorted_node_ids(set(available), rank_by_node_id)


def _enumerate_valid_solved_states(all_node_ids: Set[int], child_subquestions_by_node_id: Dict[int, Set[int]],
                                   rank_by_node_id: Dict[int, int]) -> List[Set[int]]:
    visited: Set[Tuple[int, ...]] = set()
    queue: Deque[Set[int]] = deque([set()])
    states: List[Set[int]] = []

    while queue:
        solved_node_ids = queue.popleft()
        state_key = tuple(_sorted_node_ids(solved_node_ids, rank_by_node_id))
        if state_key in visited:
            continue

        visited.add(state_key)
        states.append(set(solved_node_ids))

        for node_id in _available_nodes(all_node_ids, child_subquestions_by_node_id, solved_node_ids, rank_by_node_id):
            queue.append(set(solved_node_ids) | {node_id})

    return states


def _maximize_solved_state_without_target(initial_solved_node_ids: Set[int], target_node_id: int,
                                          all_node_ids: Set[int], child_subquestions_by_node_id: Dict[int, Set[int]],
                                          rank_by_node_id: Dict[int, int]) -> Set[int]:
    solved_node_ids = set(initial_solved_node_ids)

    while True:
        next_candidates = [
            node_id
            for node_id in _available_nodes(all_node_ids, child_subquestions_by_node_id, solved_node_ids, rank_by_node_id)
            if node_id != target_node_id
        ]
        if not next_candidates:
            break
        solved_node_ids.update(next_candidates)

    return solved_node_ids


def _build_prompt(problem: str, solved_step_statements: List[str]) -> str:
    lines = [
        "You are given a math word problem and some intermediate reasoning steps that are already solved.",
        "Your task is to produce exactly one next subquestion that should be asked to continue solving the problem.",
        "Do not solve the full problem. Do not output anything except the next subquestion.",
        "",
        f"Problem: {problem}",
        "",
        "Given that we know the answer to the following questions:",
    ]

    if solved_step_statements:
        lines.extend(solved_step_statements)
    else:
        lines.append("None.")

    lines.extend(["", "Next subquestion:"])
    return "\n".join(lines)


def _build_examples_for_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    subquestion_tree = _parse_jsonish(row.get("subquestion_tree"))
    if not isinstance(subquestion_tree, dict) or not subquestion_tree:
        return []

    question_by_node_id, child_subquestions_by_node_id, rank_by_node_id = _collect_tree_metadata(subquestion_tree)
    if not question_by_node_id:
        return []

    all_node_ids = set(question_by_node_id.keys())
    valid_states = _enumerate_valid_solved_states(all_node_ids, child_subquestions_by_node_id, rank_by_node_id)

    examples_by_key: Dict[Tuple[Tuple[int, ...], int], Dict[str, Any]] = {}

    for solved_node_ids in valid_states:
        frontier_node_ids = _available_nodes(all_node_ids, child_subquestions_by_node_id, solved_node_ids, rank_by_node_id)
        if not frontier_node_ids:
            continue

        for target_node_id in frontier_node_ids:
            unique_solved_node_ids = _maximize_solved_state_without_target(
                initial_solved_node_ids=solved_node_ids,
                target_node_id=target_node_id,
                all_node_ids=all_node_ids,
                child_subquestions_by_node_id=child_subquestions_by_node_id,
                rank_by_node_id=rank_by_node_id,
            )
            unique_frontier_node_ids = _available_nodes(
                all_node_ids,
                child_subquestions_by_node_id,
                unique_solved_node_ids,
                rank_by_node_id,
            )
            if unique_frontier_node_ids != [target_node_id]:
                continue

            ordered_solved_node_ids = _sorted_node_ids(unique_solved_node_ids, rank_by_node_id)
            solved_step_statements = [
                _normalize_question_text(question_by_node_id[node_id])
                for node_id in ordered_solved_node_ids
            ]
            state_key = tuple(ordered_solved_node_ids)
            example_key = (state_key, target_node_id)
            if example_key in examples_by_key:
                continue

            next_subquestion = question_by_node_id[target_node_id]
            examples_by_key[example_key] = {
                "solved_node_ids": ordered_solved_node_ids,
                "solved_steps": solved_step_statements,
                "next_subquestion": next_subquestion,
                "target_node_id": target_node_id,
                "prompt": _build_prompt(str(row.get("problem", "")), solved_step_statements),
            }

    return list(examples_by_key.values())


def _base_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "example_idx": row.get("example_idx"),
        "problem_type": row.get("problem_type"),
        "complexity": row.get("complexity"),
        "overlap_type": row.get("overlap_type"),
        "problem": row.get("problem"),
        "question": row.get("question"),
        "groundquery": row.get("groundquery"),
        "reasoning_trace": row.get("reasoning_trace"),
        "answer": row.get("answer"),
        "depth": row.get("depth"),
        "width": row.get("width"),
        "subquestions": row.get("subquestions"),
        "subquestions_by_node_id": row.get("subquestions_by_node_id"),
        "subquestion_tree": row.get("subquestion_tree"),
        "trees": row.get("trees"),
    }


def build_next_subquestion_datasets(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    compact_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Rows"):
        row_dict = row.to_dict()
        examples = _build_examples_for_row(row_dict)

        compact_mapping = {
            _serialize(example["solved_steps"]): example["next_subquestion"]
            for example in examples
        }

        compact_row = {
            **_base_columns(row_dict),
            COMPACT_MAPPING_COLUMN: _serialize(compact_mapping),
            "next_subquestion_examples": _serialize(examples),
            "num_next_subquestion_examples": len(examples),
        }
        compact_rows.append(compact_row)

        for pair_index, example in enumerate(examples):
            pair_rows.append(
                {
                    **_base_columns(row_dict),
                    "pair_index": pair_index,
                    "solved_node_ids": _serialize(example["solved_node_ids"]),
                    "solved_steps": _serialize(example["solved_steps"]),
                    "target_node_id": example["target_node_id"],
                    "next_subquestion": example["next_subquestion"],
                    "prompt": example["prompt"],
                }
            )

    return pd.DataFrame(compact_rows), pd.DataFrame(pair_rows)


@click.command()
@click.option("--input-path", default=str(DEFAULT_INPUT_PATH), show_default=True, help="Flattened reasoning-efficiency CSV with subquestion_tree")
@click.option("--compact-out", default=str(DEFAULT_COMPACT_OUTPUT_PATH), show_default=True, help="Output path for the compact per-problem CSV")
@click.option("--pairs-out", default=str(DEFAULT_PAIRS_OUTPUT_PATH), show_default=True, help="Output path for the exploded per-pair CSV")
@click.option("--limit", default=None, type=int, help="Optional row limit for debugging")
def cli(input_path: str, compact_out: str, pairs_out: str, limit: int):
    input_df = pd.read_csv(input_path)
    if limit is not None:
        input_df = input_df.head(limit)

    compact_df, pair_df = build_next_subquestion_datasets(input_df)

    compact_path = Path(compact_out)
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    compact_df.to_csv(compact_path, index=False)

    pairs_path = Path(pairs_out)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(pairs_path, index=False)


if __name__ == "__main__":
    cli()