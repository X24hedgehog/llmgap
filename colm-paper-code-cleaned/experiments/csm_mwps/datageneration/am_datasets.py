from collections import Counter
from typing import Dict, List, Callable
import random
import math

from mathgap.generation_util import *

from data.util import DATA_FOLDER
from datageneration.util import unique_variations_iter, generate_dataset_iter

from datageneration.arithmeticerrors import MentalModel
from trees.rules.errors import ContCompContArithError, ContTransferContArithError, ContPartWholeArithError

AM_NO_COMPEQ_RULESET = (lambda mental_model: [ContCompContArithError(mental_model), ContTransferContArithError(mental_model), ContPartWholeArithError(mental_model)])

ALL_ERRONEOUS_MENTAL_MODELS = [
    {"addition": "regrouping", "subtraction": None},
    {"addition": "forgetcarry", "subtraction": None},
    {"addition": None, "subtraction": "smallerfromlarge"},
    {"addition": "regrouping", "subtraction": "smallerfromlarge"},
    {"addition": "forgetcarry", "subtraction": "smallerfromlarge"}
]


def generate_problems_and_distractors(nr_problems: int = 1000, nr_instantiations: int = 5, vary_types: str = [PropertyType.QUANTITY],
                                      vary_formulations: bool = False, all_err_mental_models = ALL_ERRONEOUS_MENTAL_MODELS, min_depth: int = 1, expand_exp_decay_base: int = math.e, 
                                      ruleset_factory: Callable[[Dict[str, str|None]], List[InferenceRule]] = AM_NO_COMPEQ_RULESET,
                                      template_version: str = "consistent", partwhole_start_chance: float = 0.4, reduce_nonlinearity: bool = True, 
                                      offset: int = 50, leaf_min_value = 10, leaf_max_value = 25, include_rt: bool = True,
                                      nonlinearity_uniform_prob_depth: int = 4, seed: int = 14) -> Dict[int,Dict[int,Dict]]:
    """ 
        Generates the baseline dataset containing multiple numerical instantiations of a list of MWPs 
        and all answers (correct or incorrect) a student might arrive at.
        
        - nr_problems: how many MWPs should be generated
        - nr_instantiations: how many different instantiations should be generated for each MWP
        - vary_types: which property types should be varied between instantiations
        - vary_formulations: if true, formulations/templates might differ between instantiations
        - all_err_mental_models: the mental models that should be considered (e.g. the student makes a carry mistake on addition)
        - min_depth: Minimum tree depth
        - expand_exp_decay_base: base for the exponential decay of probability to continue each branch (lower value results in deeper trees)
        - ruleset_factory: factory for the set of inference rules that can be applied (takes a mental model instance)
        - template_version: name of the version holding templates for natlang rendering
        - partwhole_start_chance: chance to start with a partwhole lf instead of a container. 
            NOTE: since we're rejecting trees without ContCompCont or ContTransferCont, the actual percentage of such problems is much lower.
        - reduce_nonlinearity: will avoid expansion of ContContComp Containers where Comp is part of ContCompCompeq
        - offset: if > 0, will offset container values to make negative results less likely
        - leaf_min_value: minimum numerical value on any leaf-node (axiom)
        - leaf_max_value: maximum numerical value on any leaf-node (axiom)
        - include_rt: include the synthetic reasoning traces for the correct and all inconcsistency-based misconception answers
        - nonlinearity_uniform_prob_depth: at which depth should nonlinear rules be equally likely to be sampled compared to all other rules that are eligible
        - seed

        Returns {mwp_id: {variation_id: {problem: "", answers: [{answer: 0, proof_steps: [...]}]}}}
    """
    mental_model = MentalModel()
    ruleset = ruleset_factory(mental_model)

    random.seed(seed)
    instantiator = default_instantiator(data_folder=DATA_FOLDER, dataversion="v1", 
                                        leaf_min_value=leaf_min_value, leaf_max_value=leaf_max_value, 
                                        inner_min_value=2, inner_max_value=1_000, 
                                        strategy="cpga", max_attempts=10, validate_preselected=False)

    dataset = {}

    cons_mwps_iter = generate_dataset_iter(min_depth=min_depth, expand_exp_decay_base=expand_exp_decay_base, 
                                           ruleset=ruleset, partwhole_start_chance=partwhole_start_chance, 
                                           disallow_contcontcompcompeq_expansion=reduce_nonlinearity,
                                           nonlinearity_uniform_prob_depth=nonlinearity_uniform_prob_depth,
                                           template_version=template_version, seed=seed)
    
    # loop until we found enough mwps
    # implausible_idx = 0
    while len(dataset.keys()) < nr_problems:
        random.seed(seed)
        seed = random.randint(0, 2**32 - 1)
        mwp_id = len(dataset)
        print(mwp_id)

        # 1. generate a new mwp
        mwp = next(cons_mwps_iter)

        orig_rules_by_node = {n:n.rule for n in mwp.tree.nodes}

        orig_expression = mwp.tree.root_node.logicalform.get_quantities()[0].model_copy(deep=True)
        mwp_data = {
            "metadata": {
                "depth": mwp.tree.depth,
                "width": len(mwp.tree.leaf_nodes),
                "rule_count": dict(Counter([type(n.rule).__name__ for n in mwp.tree.nodes if not n.is_leaf])),
                "orig_rules_by_conclusion": {
                    mwp.tree.id_by_node[node]: type(orig_rules_by_node[node]).__name__ 
                    for node in mwp.tree.traverse_reasoning_trace(mwp.problem_order.body_node_ids)
                    if node.rule is not None
                },
                "depth_by_conclusion": {
                    mwp.tree.id_by_node[node]: node.depth
                    for node in mwp.tree.traverse_reasoning_trace(mwp.problem_order.body_node_ids)
                    if node.rule is not None
                },
                "correct_expression": str(orig_expression)
            },
            "instantiations": {}
        }
        
        # 2. try to sample enough variations
        try:
            mwp_var_iter = unique_variations_iter(mwp.tree, mwp.instantiation, instantiator, max_attempts_per_variation=100, vary_types=vary_types,
                                                apply_offsetting=(offset > 0), offset_min=leaf_min_value+offset, offset_max=leaf_max_value+offset, seed=seed)
            inst_id = 0
            while inst_id < nr_instantiations:
                    random.seed(seed)
                    seed = random.randint(0, 2**32 - 1)
                    
                    instantiation = next(mwp_var_iter)
                    mwp.update_instantiation(instantiation)

                    # the correct answer
                    mental_model.set_correct()
                    
                    mwp.compute_answers()
                    mwp.problem_as_nl(seed=seed)
                    mwp.reasoning_trace_as_nl(seed=seed)
                    correct_sol = mwp.numerical_answers[0]

                    mwp_data["instantiations"][inst_id] = {
                        "problem": mwp.ps_nl,
                        "instantiation": {str(k):v for k,v in mwp.instantiation.get_instantiations_of_type(PropertyType.QUANTITY).items()},
                        "correct_answer": { "answer": correct_sol },
                        "misconception_answers": [], # potential answers when only inconsistent phrasings are misunderstood
                    }
                    if include_rt:
                        mwp_data["instantiations"][inst_id]["correct_answer"]["rt"] = mwp.rt_nl

                    for mm in all_err_mental_models:
                        # set the mental model that should be used
                        mental_model.addition = mm["addition"]
                        mental_model.subtraction = mm["subtraction"]
                        
                        answer_with_model = None
                        try:
                            mwp.compute_answers()
                            answer_with_model = mwp.numerical_answers[0]
                        except ValueError as e1:
                            # if we get negative numbers, skip the mental model as its not plausible
                            if "invalid literal for int() with base 10: '-'" in str(e1):
                                continue
                            else:
                                raise e

                        if answer_with_model != correct_sol:
                            mwp.reasoning_trace_as_nl(enforce_premise_axiom_consistency=vary_formulations, seed=seed)

                            # compute the quantities on all nodes of the tree
                            all_results = [
                                quantity.eval(mwp.instantiation)
                                for node in mwp.tree.traverse_reasoning_trace(mwp.problem_order.body_node_ids)
                                for quantity in node.logicalform.get_quantities()
                            ]
                            plausible = all(r >= 2 and int(r) == r for r in all_results)

                            erroneous_option = {
                                "answer": answer_with_model,
                                "plausible": plausible,
                                "misconceptions": list(filter(lambda x: x is not None, [mental_model.addition, mental_model.subtraction]))
                            }
                            if include_rt:
                                erroneous_option["rt"] = mwp.rt_nl

                            mwp_data["instantiations"][inst_id]["misconception_answers"].append(erroneous_option)
                    
                    if len(mwp_data["instantiations"][inst_id]["misconception_answers"]) > 0:
                        inst_id += 1
                
        except ValueError as e:
            # if it takes too long to find enough valid instantiations, simply sample another mwp
            if "Failed to find a valid instantiation" in str(e):
                continue
            else:
                raise e    
        
        dataset[mwp_id] = mwp_data    
    return dataset