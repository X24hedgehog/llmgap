import json
import random
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

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


def _load_existing_csv_state(out_file: Path) -> Tuple[Set[str], int, bool]:
    if not out_file.exists() or out_file.stat().st_size == 0:
        return set(), 0, False

    existing_df = pd.read_csv(out_file, usecols=["example_idx", "problem"])
    if existing_df.empty:
        return set(), 0, False

    existing_problem_texts = set(existing_df["problem"].dropna().astype(str))
    next_example_idx = int(existing_df["example_idx"].max()) + 1
    return existing_problem_texts, next_example_idx, True


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


def all_variants_for_problem(example_idx: int, problem: Dict) -> List[Dict]:
    """Return all ~31 variant rows for a single problem (same as generate_csv.py)."""
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
                row_problem_type = "control" if overlap_type == "control" else variant_group
                rows.append(
                    _make_row(
                        example_idx=example_idx,
                        variant_group=row_problem_type,
                        complexity=complexity,
                        overlap_type=overlap_type,
                        payload=payload,
                        gold_subquestions=base_gold,
                    )
                )

    return rows


def pick_one_variant(example_idx: int, problem: Dict, rng: random.Random) -> Dict:
    """Randomly pick exactly 1 of the ~31 variants for this problem."""
    variants = all_variants_for_problem(example_idx, problem)
    return rng.choice(variants)


def write_csv_dataset_in_batches(out_path: str, nr_problems: int, min_depth: int, max_depth: int,
                                 num_irrelevant_trees: int, profile: str = "full", data_folder: str = None,
                                 seed: int = 14, max_attempts_per_problem: int = 100,
                                 checkpoint_base_problems: int = CHECKPOINT_BASE_PROBLEMS) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    existing_problem_texts, example_idx_offset, wrote_header = _load_existing_csv_state(out_file)
    remaining_problems = nr_problems - len(existing_problem_texts)
    if remaining_problems <= 0:
        click.echo(f"CSV already has {len(existing_problem_texts)} rows; nothing to add.")
        return

    seed_offset = len(existing_problem_texts)
    rng = random.Random(seed + seed_offset)
    written_new_rows = 0
    batch_seed_offset = 0
    empty_batches = 0

    progress_bar = tqdm(total=remaining_problems, desc="CSV batches")
    while written_new_rows < remaining_problems:
        batch_size = min(checkpoint_base_problems, remaining_problems - written_new_rows)
        batch_dataset = generate_dataset(
            nr_problems=batch_size,
            min_depth=min_depth,
            max_depth=max_depth,
            num_irrelevant_trees=num_irrelevant_trees,
            profile=profile,
            data_folder=data_folder,
            seed=seed + seed_offset + batch_seed_offset,
            max_attempts_per_problem=max_attempts_per_problem,
        )
        batch_seed_offset += checkpoint_base_problems

        batch_rows = []
        for problem in batch_dataset:
            row = pick_one_variant(example_idx_offset, problem, rng)
            if row["problem"] in existing_problem_texts:
                continue

            existing_problem_texts.add(row["problem"])
            batch_rows.append(row)
            example_idx_offset += 1
            written_new_rows += 1

        if batch_rows:
            batch_df = pd.DataFrame(batch_rows)
            batch_df.to_csv(out_file, mode="a", header=not wrote_header, index=False)
            wrote_header = True
            progress_bar.update(len(batch_rows))
            empty_batches = 0
        else:
            empty_batches += 1
            if empty_batches >= 20:
                progress_bar.close()
                raise RuntimeError("Unable to find new problems after 20 consecutive batches.")

    progress_bar.close()


@click.command()
@click.option("--out-path", "out_path", required=True, help="Where the generated CSV will be stored")
@click.option("--nr-problems", default=2000, show_default=True, help="The number of base problems to generate")
@click.option("--min-depth", default=3, show_default=True, help="Minimum proof depth")
@click.option("--max-depth", default=4, show_default=True, help="Maximum proof depth")
@click.option("--num-irrelevant-trees", default=4, show_default=True, help="Number of irrelevant trees to attach")
@click.option("--profile", type=click.Choice(["comparison", "full"]), default="full", show_default=True, help="Generator profile")
@click.option("--seed", default=42, show_default=True, help="Random seed")
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
