from typing import Dict, List, Set, Type, Generator as GeneratorType
import random
from itertools import product
import math

from mathgap.expressions import Expr
from mathgap.instantiate.instantiation import Instantiation, delete_all_of_type
from mathgap.instantiate.instantiators import Instantiator
from mathgap.logicalforms.container import Container
from mathgap.mathwordproblems import MathWordProblem
from mathgap.properties import PropertyKey, PropertyType
from mathgap.trees.prooftree import ProofTree, TraversalOrder, TreeNode
from mathgap.trees.rules.inference_rule import InferenceRule

from mathgap.natlang.templates.template import WHITESPACE
from mathgap.trees.generators import MultiGenerator
from mathgap.trees.generators.stoppingcriteria.logic import OrCriterion
from mathgap.generation_util import *

from data.util import DATA_FOLDER
from trees.generators.general import GeneralMisconGenerator
from trees.generators.stoppingcriteria.limitednonlinearity import AvoidContContCompOfCompeqExpansion

from datageneration.stoppingcriteria import ProbabilisticBranchDepthCriterion
from datageneration.rulesamplingpolicies import NonlinearMoreLikelyWithDepthPolicy

def iter_all_errors_rec(mwp: MathWordProblem, conclusions: List[TreeNode], 
                        original_rules_by_node: Dict[TreeNode, InferenceRule],
                        errors_by_type: Dict[Type, List[InferenceRule]]) -> GeneratorType[MathWordProblem, None, None]:
    """ 
        Iterates over all conclusions and for each one of them applies the original rule and all errors.

        This way we can get all potential wrong "paths" a student might take, 
        assuming that the student can only make mistakes on conclusions
        and the set of mistakes is limited to errors_by_type.

        NOTE: the function modifies a single instance of a tree for speed, so DO NOT USE any reference to the tree afterwards 
              but DIRECTLY EXTRACT DATA from the tree instead!
    """
    if len(conclusions) == 0:
        mwp.tree.compute_symbolically()
        mwp.compute_answers()
        yield mwp
    else:
        potentially_wrong_conclusion = conclusions[0]
        original_rule_for_conclusion = original_rules_by_node[potentially_wrong_conclusion]

        # use the correct inference rule for this conclusion
        potentially_wrong_conclusion.rule = original_rule_for_conclusion
        
        for v in iter_all_errors_rec(mwp, conclusions[1:], original_rules_by_node, errors_by_type):
            if v is None: break
            yield v

        # try all incorrect inference rules / misconceptions applicable on this conclusion
        for error_for_type in errors_by_type.get(type(original_rule_for_conclusion), []):
            potentially_wrong_conclusion.rule = error_for_type

            for v in iter_all_errors_rec(mwp, conclusions[1:], original_rules_by_node, errors_by_type):
                if v is None: break
                yield v

    yield None


def compute_all_op_pert_erroneous_answers(e: Expr|str, instantiation: Instantiation|Dict) -> List[float]:
    """ 
        Computes all possible operation perturbations (e.g. + becomes -) and the resulting answer,
        except for the original answer.
    """
    if isinstance(instantiation, Instantiation):
        vars = {str(k):v for k,v in instantiation.get_instantiations_of_type(PropertyType.QUANTITY).items()}
    else:
        vars = instantiation.copy()

    if isinstance(e, Expr):
        orig_expr = str(e)
    else:
        orig_expr = e

    add_indices = [i for i,char in enumerate(orig_expr) if char in set(["+", "-"])]
    combinations = product(["+", "-"], repeat=len(add_indices))

    answers = []
    for combo in combinations:
        new_expr = orig_expr + ""
        for op,index in zip(combo, add_indices):
            new_expr = new_expr[:index] + op + new_expr[index+1:]
        
        if new_expr != orig_expr:
            answers.append(eval(new_expr, vars))

    return answers

def unique_variations_iter(tree: ProofTree, orig_instantiation: Instantiation, instantiator: Instantiator, max_attempts_per_variation: int = 100, 
                           vary_types: List[PropertyType] = [PropertyType.QUANTITY], apply_offsetting: bool = True, offset_min: int = 100, offset_max: int = 200,
                           seed: int = 14) -> GeneratorType[Instantiation, None, None]:
    """ 
        Iteratively creates new unique numerical variations 
        NOTE: will become slower the more unique variations are drawn

        - vary_types: which types of properties will be varied
        - max_attempts_per_variation: maximum number of tries to find a new valid instantiation, if exceeded, an error is thrown
        - apply_offsetting: if true, then containers will be offset such that the value is in range
            - offset_min
            - offset_max
    """
    used_instantiations = []

    attempts_since_last_success = 0
    while True:
        attempts_since_last_success += 1
        seed += 1
        instantiation = orig_instantiation
        for typ in vary_types:
            instantiation = delete_all_of_type(instantiation, typ)

        # Offsetting
        if apply_offsetting:
            offset_quantities: List[PropertyKey] = []
            inc_node_ids = set([tree.id_by_node[tree.root_node]])
            for node in tree.traverse(TraversalOrder.DFS):
                if not tree.id_by_node[node] in inc_node_ids: continue

                if node.is_leaf:
                    offset_quantities.extend([var.identifier for var in node.logicalform.get_quantities()])

                for child_node in node.child_nodes:
                    if isinstance(child_node.logicalform, Container):
                        inc_node_ids.add(tree.id_by_node[child_node])
                
            offset_quantities = set(offset_quantities)
            
            for q in offset_quantities:
                instantiation[q] = random.randrange(offset_min, offset_max)

        try:
            instantiation = instantiator.instantiate(tree, instantiation, skip_existing=True, seed=seed)

            if not any(i == instantiation for i in used_instantiations):
                attempts_since_last_success = 0
                used_instantiations.append(instantiation)
                yield instantiation
        except ValueError as e:
            # once we exceed the max nr of tries to find a new variation, we forward the error
            if attempts_since_last_success > max_attempts_per_variation:
                raise e
        
        if attempts_since_last_success > max_attempts_per_variation:
            raise ValueError(f"Failed to find a new instantiation that isn't already present!")
            
def choose_potentially_misconcievable_conclusions(tree: ProofTree, selectable_rule_types: Set[Type], 
                                                  prob_select: float = 0.5, max_attempts: int = 100, seed: int = 14) -> List[TreeNode]:
    """ Selects a non-empty random subset of conclusions that are derived by rules for which we could have misconceptions """
    random.seed(seed)
    applicable_nodes = [node for node in tree.traverse() if type(node.rule) in selectable_rule_types]
    
    if len(applicable_nodes) == 0: 
        raise ValueError("Need at least one conclusion where a misconception is applicable to choose a non-empty set of nodes!")

    for _ in range(max_attempts):
        selected_conclusions = []
        
        for node in tree.traverse():
            rule_type = type(node.rule)
            if rule_type in selectable_rule_types:
                if random.random() <= prob_select:
                    selected_conclusions.append(node)
        
        if len(selected_conclusions) > 0:
            return selected_conclusions

    raise ValueError("Failed to find a non-empty subset of conclusions which are misconcievable!")

def generate_dataset_iter(min_depth: int = 1, expand_exp_decay_base: int = math.e, ruleset: List[InferenceRule] = [], partwhole_start_chance: float = 0.5,
                          disallow_contcontcompcompeq_expansion: bool = True, nonlinearity_uniform_prob_depth: int = 4, template_version: str = "consistent", seed: int = 14) -> GeneratorType[MathWordProblem, None, None]:
    """ 
        Generates a dataset of mwps, where the underlying proof tree is of depth at least of min_depth and then deeper mwps are exponentially less likely
        
        - min_depth: minimum depth the generated tree will have
        - expand_exp_decay_base: base for the exponential decay of probability to continue each branch (lower value results in deeper trees)
        - nonlinearity_uniform_prob_depth: at which depth should nonlinear rules be equally likely to be sampled compared to all other rules that are eligible
        - disallow_contcontcompcompeq_expansion: will avoid expanding Containers of ContContComp that is used for a ContCompCompeq
        - template_version: which templates should be used by default

        NOTE: this might take some time depending on your settings 
        (usually finding a valid instantiation for the numbers takes the longest in complex trees)
    """

    # 1. Define the generators for generating the proof-trees
    prob_depth_crit = ProbabilisticBranchDepthCriterion(min_depth, base=expand_exp_decay_base)
    non_linear_rules = [r for r in ruleset if type(r) == ContCompCompeqCont]
    rule_sampling_policy = NonlinearMoreLikelyWithDepthPolicy(start_depth=0, uniform_depth=nonlinearity_uniform_prob_depth, non_linear_rules=non_linear_rules)
    stopping_criterion = OrCriterion([prob_depth_crit, AvoidContContCompOfCompeqExpansion()]) if disallow_contcontcompcompeq_expansion else prob_depth_crit
    weights_by_generator = {
        GeneralMisconGenerator(use_attribute=use_attrib, use_unit=False, 
            comp_same_entity_prob=1.0, compeq_same_entity_prob=1.0, 
            max_part_whole=3, 
            stopping_criterion=stopping_criterion, 
            start_types=[start_type], 
            inference_rules=ruleset,
            rule_sampling_policy=rule_sampling_policy,
            comp_allowed_comparisons=ADDITIVE_COMP_TYPES): (1.0 - partwhole_start_chance) if start_type == Container else partwhole_start_chance
        for use_attrib in [True, False]
        for start_type in [Container, PartWhole]
    }
    generator = MultiGenerator(weights_by_generator)

    # 2. Load the default instantiator but use the agents, entities etc specified in the data-folder of this experiment
    #    The instantiator will be used to instantiate properties with values (e.g. agent1 -> Alice, quantity1 -> 4)
    instantiator = default_instantiator(data_folder=DATA_FOLDER, dataversion="v1", leaf_min_value=2, leaf_max_value=25, 
                                        inner_min_value=2, inner_max_value=1_000, max_attempts=10_000, strategy="cpga")
    
    # 3. Load template renderers and samplers
    #    They will be used to express logical forms and deduction steps as natural language
    ps_template_sampler, ps_answers_template_sampler, ps_renderer, rt_template_sampler, rt_renderer \
        = default_templates_and_samplers(DATA_FOLDER, template_version, WHITESPACE)

    # 4. Return the parameterized-iterator that can generate mwps
    return generate_mwps_iter(generator, instantiator, CANONICAL_ORDER_SAMPLER, 
                              ps_template_sampler, ps_answers_template_sampler, ps_renderer, 
                              rt_template_sampler, rt_renderer, seed)