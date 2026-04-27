import json
from pathlib import Path
import sys
from typing import Dict, List

import click
import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate import generate_dataset


CHECKPOINT_BASE_PROBLEMS = 5


def _join_problem_text(problembody: List[Dict], question: Dict) -> str:
    body_text = " ".join(sentence["text"] for sentence in problembody)
    return f"{body_text} {question['text']}".strip()


def _join_reasoning_trace(rt: List[Dict]) -> str:
    return " ".join(step["text"] for step in rt)


def _serialize(value):
    if value is None:
        return ""
    return json.dumps(value)


def _extract_base_gold(problem: Dict) -> Dict:
    base = problem["base"]
    return {
        "subquestions": base.get("subquestions", []),
        "subquestions_by_node_id": base.get("subquestions_by_node_id", {}),
        "subquestion_answers_by_node_id": base.get("subquestion_answers_by_node_id", {}),
        "subproblems_by_node_id": base.get("subproblems_by_node_id", {}),
        "subquestion_tree": base.get("subquestion_tree"),
    }


def _make_row(example_idx: int, variant_group: str, complexity: str, overlap_type: str,
              payload: Dict, gold_subquestions: Dict) -> Dict:
    base_tree = payload.get("trees", {}).get("base", {})

    return {
        "example_idx": example_idx,
        "problem_type": variant_group,
        "complexity": complexity,
        "overlap_type": overlap_type,
        "problem": _join_problem_text(payload["problembody"], payload["question"]),
        "question": payload["question"]["text"],
        "groundquery": payload["groundquery"]["text"],
        "reasoning_trace": _join_reasoning_trace(payload["rt"]),
        "answer": payload["answer"],
        "depth": base_tree.get("depth", ""),
        "width": base_tree.get("width", ""),
        "subquestions": _serialize(gold_subquestions.get("subquestions", [])),
        "subquestions_by_node_id": _serialize(gold_subquestions.get("subquestions_by_node_id", {})),
        "subquestion_answers_by_node_id": _serialize(gold_subquestions.get("subquestion_answers_by_node_id", {})),
        "subproblems_by_node_id": _serialize(gold_subquestions.get("subproblems_by_node_id", {})),
        "subquestion_tree": _serialize(gold_subquestions.get("subquestion_tree")),
        "trees": _serialize(payload.get("trees", {})),
    }


def flatten_problem(example_idx: int, problem: Dict) -> List[Dict]:
    rows = []
    base_gold = _extract_base_gold(problem)

    rows.append(
        _make_row(
            example_idx=example_idx,
            variant_group="base",
            complexity="base",
            overlap_type="base",
            payload=problem["base"],
            gold_subquestions=base_gold,
        )
    )

    for variant_group in ["connected", "disconnected"]:
        for complexity, variants in problem[variant_group].items():
            for overlap_type, payload in variants.items():
                if overlap_type == "control":
                    gold_subquestions = base_gold
                    row_problem_type = "control"
                else:
                    gold_subquestions = base_gold
                    row_problem_type = variant_group

                rows.append(
                    _make_row(
                        example_idx=example_idx,
                        variant_group=row_problem_type,
                        complexity=complexity,
                        overlap_type=overlap_type,
                        payload=payload,
                        gold_subquestions=gold_subquestions,
                    )
                )

    return rows


def generate_csv_dataset(nr_problems: int, min_depth: int, max_depth: int, num_irrelevant_trees: int,
                         profile: str = "full", data_folder: str = None, seed: int = 14,
                         max_attempts_per_problem: int = 100) -> pd.DataFrame:
    dataset = generate_dataset(
        nr_problems=nr_problems,
        min_depth=min_depth,
        max_depth=max_depth,
        num_irrelevant_trees=num_irrelevant_trees,
        profile=profile,
        data_folder=data_folder,
        seed=seed,
        max_attempts_per_problem=max_attempts_per_problem,
    )

    rows = []
    for example_idx, problem in enumerate(dataset):
        rows.extend(flatten_problem(example_idx, problem))

    return pd.DataFrame(rows)


def write_csv_dataset_in_batches(out_path: str, nr_problems: int, min_depth: int, max_depth: int,
                                 num_irrelevant_trees: int, profile: str = "full", data_folder: str = None,
                                 seed: int = 14, max_attempts_per_problem: int = 100,
                                 checkpoint_base_problems: int = CHECKPOINT_BASE_PROBLEMS) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    example_idx_offset = 0
    wrote_header = False

    batch_starts = range(0, nr_problems, checkpoint_base_problems)
    for batch_start in tqdm(batch_starts, desc="CSV batches"):
        batch_size = min(checkpoint_base_problems, nr_problems - batch_start)
        batch_dataset = generate_dataset(
            nr_problems=batch_size,
            min_depth=min_depth,
            max_depth=max_depth,
            num_irrelevant_trees=num_irrelevant_trees,
            profile=profile,
            data_folder=data_folder,
            seed=seed + batch_start,
            max_attempts_per_problem=max_attempts_per_problem,
        )

        batch_rows = []
        for batch_example_idx, problem in enumerate(batch_dataset):
            batch_rows.extend(flatten_problem(example_idx_offset + batch_example_idx, problem))

        batch_df = pd.DataFrame(batch_rows)
        batch_df.to_csv(out_file, mode="a", header=not wrote_header, index=False)

        wrote_header = True
        example_idx_offset += batch_size


@click.command()
@click.option("--out-path", "out_path", required=True, help="Where the generated CSV will be stored")
@click.option("--nr-problems", default=50, show_default=True, help="The number of problems to generate")
@click.option("--min-depth", default=3, show_default=True, help="Minimum proof depth")
@click.option("--max-depth", default=4, show_default=True, help="Maximum proof depth")
@click.option("--num-irrelevant-trees", default=4, show_default=True, help="Number of irrelevant trees to attach")
@click.option("--profile", type=click.Choice(["comparison", "full"]), default="full", show_default=True, help="Generator profile")
@click.option("--seed", default=14, show_default=True, help="Random seed")
@click.option("--data-folder", default=None, help="Override the proof_search data folder")
@click.option("--max-attempts-per-problem", default=100, show_default=True, help="How many retries to allow for each sample")
def cli(out_path: str, nr_problems: int, min_depth: int, max_depth: int, num_irrelevant_trees: int,
        profile: str, seed: int, data_folder: str, max_attempts_per_problem: int):
    write_csv_dataset_in_batches(
        out_path=out_path,
        nr_problems=nr_problems,
        min_depth=min_depth,
        max_depth=max_depth,
        num_irrelevant_trees=num_irrelevant_trees,
        profile=profile,
        data_folder=data_folder,
        seed=seed,
        max_attempts_per_problem=max_attempts_per_problem,
    )


if __name__ == "__main__":
    cli()