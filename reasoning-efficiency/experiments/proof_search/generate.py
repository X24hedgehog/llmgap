import json
import os
import random
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import click
import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mathgap.data.util import load_entities
from mathgap.generation_util import *
from mathgap.natlang.templates.template import WHITESPACE
from mathgap.natlang.templates.templaterenderer import TemplateRenderer
from mathgap.trees.generators import MultiGenerator

from generationutil import get_next_problem


RETRYABLE_ERROR_SNIPPETS = (
    "Failed to find a valid instantiation after",
    "Invalid whole <-> part mapping",
    "Base tree does not have enough",
    "not a valid super-entity",
    "Requires at least one super-entity with sub-entity",
    "Can only have one pre-instantiated part of a whole if we need to resort to whole=part",
    "Cannot overwrite",
)


def resolve_data_folder() -> str:
    return str(Path(__file__).resolve().parent / "data")


def load_entities_by_topic(data_folder: str) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
    topic_path = Path(data_folder) / "entities" / "extended" / "entities_topic.json"
    with open(topic_path, "r") as handle:
        return {
            tuple(topic_data["topic"]): [tuple(entity) for entity in topic_data["entities"]]
            for topic_data in json.load(handle)
        }


def load_lexical_overlaps(data_folder: str) -> List[str]:
    overlap_path = Path(data_folder) / "agents" / "extended" / "lexical_overlap.csv"
    return list(pd.read_csv(overlap_path, index_col=False, names=["name"])["name"])


def build_topic_entities(data_folder: str, entities_by_topic: Dict[Tuple[str, str], List[Tuple[str, str]]]) -> Dict:
    topic_entities = load_entities(data_folder=data_folder, version="extended")
    all_entities_with_topic = {
        entity
        for entities in entities_by_topic.values()
        for entity in entities
    }
    topic_entities["entities_without_units"] = [
        entity
        for entity in topic_entities["entities_without_units"]
        if entity in all_entities_with_topic
    ]
    topic_entities["parts_by_whole"] = {
        whole: [part for part in parts if part in all_entities_with_topic]
        for whole, parts in topic_entities["parts_by_whole"].items()
    }
    topic_entities["super_sub_entities"] = {
        whole: [part for part in parts if part in all_entities_with_topic]
        for whole, parts in topic_entities["super_sub_entities"].items()
    }
    return topic_entities


def build_generators(profile: str, min_depth: int, max_depth: int) -> Tuple[MultiGenerator, MultiGenerator]:
    if profile == "comparison":
        weights_by_generator_base = {
            default_generator(
                use_attribute=False,
                use_unit=False,
                comp_same_entity_prob=1.0,
                compeq_same_entity_prob=1.0,
                stopping_criterion=BranchDepthCriterion(depth),
                start_types=CONT_START_TYPE,
                inference_rules=COMP_RULESET,
            ): 1.0
            for depth in range(min_depth, max_depth + 1)
        }
        base_generator = MultiGenerator(weights_by_generator_base)
        irrelevant_generator = base_generator
        return base_generator, irrelevant_generator

    ruleset = [
        ContTransferCont(),
        ContCompCont(),
        ContCompCompeqCont(),
        ContContComp(),
        ContPartWhole(),
        ContRateCont(),
    ]
    irrelevant_ruleset = [
        ContTransferCont(),
        ContCompCont(),
        ContContComp(),
        ContPartWhole(),
    ]

    weights_by_generator_base = {
        default_generator(
            use_attribute=False,
            use_unit=False,
            comp_same_entity_prob=1.0,
            compeq_same_entity_prob=1.0,
            stopping_criterion=BranchDepthCriterion(depth),
            start_types=start_type,
            inference_rules=ruleset,
        ): 1.0
        for depth in range(min_depth, max_depth + 1)
        for start_type in [CONT_START_TYPE, PARTWHOLE_START_TYPE]
    }
    weights_by_generator_irrelevant = {
        default_generator(
            use_attribute=False,
            use_unit=False,
            comp_same_entity_prob=1.0,
            compeq_same_entity_prob=1.0,
            stopping_criterion=BranchDepthCriterion(depth),
            start_types=CONT_START_TYPE,
            inference_rules=irrelevant_ruleset,
        ): 1.0
        for depth in range(min_depth, max_depth + 1)
    }

    return MultiGenerator(weights_by_generator_base), MultiGenerator(weights_by_generator_irrelevant)


def build_generation_components(profile: str, data_folder: str, min_depth: int, max_depth: int):
    entities_by_topic = load_entities_by_topic(data_folder)
    lexical_overlaps = load_lexical_overlaps(data_folder)
    topic_entities = build_topic_entities(data_folder, entities_by_topic)
    base_generator, irrelevant_generator = build_generators(profile, min_depth, max_depth)

    instantiator = default_instantiator(
        data_folder=data_folder,
        dataversion="extended",
        leaf_min_value=2,
        leaf_max_value=99,
        inner_min_value=2,
        inner_max_value=10_000,
        max_attempts=1_000,
        strategy="cpga",
    )
    topic_instantiator = default_instantiator(
        data_folder=data_folder,
        dataversion="extended",
        leaf_min_value=2,
        leaf_max_value=20,
        inner_min_value=2,
        inner_max_value=1_000,
        entities=topic_entities,
        strategy="cpga",
    )

    problem_order_sampler = CanonicalOrderSampler()
    template_renderer = TemplateRenderer()
    ps_template_sampler, _, ps_renderer, rt_template_sampler, rt_renderer = default_templates_and_samplers(
        data_folder,
        "extended",
        WHITESPACE,
    )

    return {
        "base_generator": base_generator,
        "irrelevant_generator": irrelevant_generator,
        "instantiator": instantiator,
        "topic_instantiator": topic_instantiator,
        "entities_by_topic": entities_by_topic,
        "lexical_overlaps": lexical_overlaps,
        "problem_order_sampler": problem_order_sampler,
        "ps_template_sampler": ps_template_sampler,
        "ps_renderer": ps_renderer,
        "rt_template_sampler": rt_template_sampler,
        "rt_renderer": rt_renderer,
        "template_renderer": template_renderer,
    }


def is_retryable_error(error: Exception) -> bool:
    if isinstance(error, (AssertionError, IndexError, NotImplementedError, ValueError)):
        return True

    message = str(error)
    return any(snippet in message for snippet in RETRYABLE_ERROR_SNIPPETS)


def generate_dataset(nr_problems: int, min_depth: int, max_depth: int, num_irrelevant_trees: int,
                     profile: str = "full", data_folder: str = None, seed: int = 14,
                     max_attempts_per_problem: int = 100) -> List[Dict]:
    data_folder = resolve_data_folder() if data_folder is None else data_folder
    components = build_generation_components(profile, data_folder, min_depth, max_depth)

    random.seed(seed)
    dataset = []

    for _ in tqdm(range(nr_problems), desc="Generating problems"):
        problem = None
        last_error = None

        for _attempt in range(max_attempts_per_problem):
            sample_seed = random.randint(0, 2**32 - 1)
            try:
                problem = get_next_problem(
                    components["base_generator"],
                    components["irrelevant_generator"],
                    components["instantiator"],
                    components["topic_instantiator"],
                    components["entities_by_topic"],
                    components["lexical_overlaps"],
                    num_irrelevant_trees,
                    components["problem_order_sampler"],
                    components["ps_template_sampler"],
                    components["ps_renderer"],
                    components["rt_template_sampler"],
                    components["rt_renderer"],
                    components["template_renderer"],
                    seed=sample_seed,
                )
                break
            except Exception as error:
                last_error = error
                if not is_retryable_error(error):
                    raise

        if problem is None:
            raise RuntimeError(
                f"Failed to generate problem after {max_attempts_per_problem} attempts"
            ) from last_error

        dataset.append(problem)

    return dataset


def save_dataset(dataset: List[Dict], out_path: str) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.suffix.lower() == ".jsonl":
        with open(out_file, "w") as handle:
            for row in dataset:
                handle.write(json.dumps(row))
                handle.write("\n")
        return

    with open(out_file, "w") as handle:
        json.dump(dataset, handle, indent=2)


@click.command()
@click.option("--out-path", "out_path", required=True, help="Where the generated dataset will be stored (.json or .jsonl)")
@click.option("--nr-problems", default=50, show_default=True, help="The number of problems to generate")
@click.option("--min-depth", default=2, show_default=True, help="Minimum proof depth")
@click.option("--max-depth", default=3, show_default=True, help="Maximum proof depth")
@click.option("--num-irrelevant-trees", default=3, show_default=True, help="Number of irrelevant trees to attach")
@click.option("--profile", type=click.Choice(["comparison", "full"]), default="full", show_default=True, help="Generator profile")
@click.option("--seed", default=14, show_default=True, help="Random seed")
@click.option("--data-folder", default=None, help="Override the proof_search data folder")
@click.option("--max-attempts-per-problem", default=100, show_default=True, help="How many retries to allow for each sample")
def cli(out_path: str, nr_problems: int, min_depth: int, max_depth: int, num_irrelevant_trees: int,
        profile: str, seed: int, data_folder: str, max_attempts_per_problem: int):
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
    save_dataset(dataset, out_path)


if __name__ == "__main__":
    cli()