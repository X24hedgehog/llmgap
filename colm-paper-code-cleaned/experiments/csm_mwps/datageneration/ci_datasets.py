from collections import Counter
from typing import Dict, List, Set, Type, Generator as GeneratorType
import random
import math
from datetime import datetime

from mathgap.instantiate.instantiation import delete_all_of_type
from mathgap.logicalforms import LogicalForm, ALL_COMP_TYPES, ALL_COMP_TYPES_EXCEPT_DIVISION
from mathgap.natlang.templates.template import WHITESPACE
from mathgap.trees.generators import MultiGenerator
from mathgap.trees.generators.stoppingcriteria.logic import OrCriterion
from mathgap.trees.generators.stoppingcriteria.treewidth import TreeWidthCriterion
from mathgap.trees.prooftree import TreeNode
from mathgap.trees.sampling.order import OrderSampler
from mathgap.generation_util import *

from data.util import DATA_FOLDER
from datageneration.util import choose_potentially_misconcievable_conclusions, compute_all_op_pert_erroneous_answers, generate_dataset_iter, iter_all_errors_rec, unique_variations_iter
from trees.rules.misconceptions import ContTransferContMisconceptionIncons, ContCompContMisconceptionIncons
from trees.generators.stoppingcriteria.limitednonlinearity import AvoidContContCompOfCompeqExpansion

from datageneration.stoppingcriteria import ProbabilisticBranchDepthCriterion
from datageneration.rulesamplingpolicies import NonlinearMoreLikelyWithDepthPolicy

FULL_MISCONCEPTIONS_BY_RULE_TYPE = {
    ContTransferCont: [ContTransferContMisconceptionIncons()],
    ContCompCont: [ContCompContMisconceptionIncons()],
}

COMP_MISCONCEPTIONS_BY_RULE_TYPE = {
    ContCompCont: [ContCompContMisconceptionIncons()],
}

FULL_RULESET = [ContCompCont(), ContTransferCont(), ContPartWhole(), ContCompCompeqCont(), ContContComp()]
NO_COMPEQ_RULESET = [ContCompCont(), ContTransferCont(), ContPartWhole()]
COMP_RULESET = [ContCompCont()]
TRANSFER_PARTWHOLE_RULESET = [ContTransferCont(), ContPartWhole()]

def generate_problems_and_distractors(nr_problems: int = 1000, nr_instantiations: int = 5, vary_types: str = [PropertyType.QUANTITY],
                                      vary_formulations: bool = False, vary_applicable: bool = False, prob_misconcievable: float = 0.5, 
                                      min_depth: int = 1, expand_exp_decay_base: int = math.e, 
                                      ruleset: List[InferenceRule] = FULL_RULESET, miscon_ruleset: Dict[Type, List[InferenceRule]] = FULL_MISCONCEPTIONS_BY_RULE_TYPE,
                                      cons_template_version: str = "consistent", incons_template_version: str = "inconsistent",
                                      partwhole_start_chance: float = 0.4, reduce_nonlinearity: bool = True, 
                                      offset: int = 50, leaf_min_value = 2, leaf_max_value = 25, include_op_pert: bool = False, include_rt: bool = False,
                                      include_cons_form: bool = False, nonlinearity_uniform_prob_depth: int = 4, omit_non_applicable: bool = True, seed: int = 14) -> Dict[int,Dict[int,Dict]]:
    """ 
        Generates the baseline dataset containing multiple numerical instantiations of a list of MWPs 
        with applicable inconsistency errors and all answers (correct or incorrect) a student might arrive at.
        
        - nr_problems: how many MWPs should be generated
        - nr_instantiations: how many different instantiations should be generated for each MWP
        - vary_types: which property types should be varied between instantiations
        - vary_formulations: if true, formulations/templates might differ between instantiations
        - vary_applicable: if true, nodes selected as inconsistent might vary between instantiations
        - prob_misconcievable: the probability for each rule to be reformulated in a misconcievable manner (inconsistent phrasing)
        - min_depth: Minimum tree depth
        - expand_exp_decay_base: base for the exponential decay of probability to continue each branch (lower value results in deeper trees)
        - ruleset: the set of inference rules that can be applied
        - miscon_ruleset: defines the erroneous rules that can be applied for each conclusion type
        - cons_template_version: name of the version holding consistent templates for natlang rendering
        - incons_template_version: name of the verison holding inconsistent templates for natlang rendering
        - partwhole_start_chance: chance to start with a partwhole lf instead of a container. 
            NOTE: since we're rejecting trees without ContCompCont or ContTransferCont, the actual percentage of such problems is much lower.
        - reduce_nonlinearity: will avoid expansion of ContContComp Containers where Comp is part of ContCompCompeq
        - offset: if > 0, will offset container values to make negative results less likely
        - leaf_min_value: minimum numerical value on any leaf-node (axiom)
        - leaf_max_value: maximum numerical value on any leaf-node (axiom)
        - include_op_pert: include all operation perturbations
        - include_rt: include the synthetic reasoning traces for the correct and all inconcsistency-based misconception answers
        - include_cons_form: include the consistent form of the MWP where none of the inconsistency errors are applicable
        - nonlinearity_uniform_prob_depth: at which depth should nonlinear rules be equally likely to be sampled compared to all other rules that are eligible
        - omit_non_applicable: omit problems where no distractor is applicable
        - seed

        Returns {mwp_id: {variation_id: {problem: "", answers: [{answer: 0, proof_steps: [...]}]}}}
    """
    random.seed(seed)
    instantiator = default_instantiator(data_folder=DATA_FOLDER, dataversion="v1", 
                                        leaf_min_value=leaf_min_value, leaf_max_value=leaf_max_value, 
                                        inner_min_value=2, inner_max_value=1_000, 
                                        strategy="cpga", max_attempts=10, validate_preselected=False)
    inconsistent_sampler = TemplateSampler(load_templates(data_folder=DATA_FOLDER, version=incons_template_version))

    dataset = {}

    cons_mwps_iter = generate_dataset_iter(min_depth=min_depth, expand_exp_decay_base=expand_exp_decay_base, 
                                           ruleset=ruleset, partwhole_start_chance=partwhole_start_chance, 
                                           disallow_contcontcompcompeq_expansion=reduce_nonlinearity,
                                           nonlinearity_uniform_prob_depth=nonlinearity_uniform_prob_depth,
                                           template_version=cons_template_version, seed=seed)
    
    # loop until we found enough mwps
    # implausible_idx = 0
    while len(dataset.keys()) < nr_problems:
        random.seed(seed)
        seed = random.randint(0, 2**32 - 1)
        mwp_id = len(dataset)

        if mwp_id % 100 == 0: 
            print(f"{datetime.now()}: {mwp_id}")

        # 1. generate a new mwp
        mwp = next(cons_mwps_iter)
        
        # 2. try to sample enough variations
        try:
            mwp_var_iter = unique_variations_iter(mwp.tree, mwp.instantiation, instantiator, max_attempts_per_variation=100, vary_types=vary_types,
                                                  apply_offsetting=(offset > 0), offset_min=leaf_min_value+offset, offset_max=leaf_max_value+offset, seed=seed)
            variations = [next(mwp_var_iter) for i in range(nr_instantiations)]
        except ValueError as e:
            # if it takes too long to find enough valid instantiations, simply sample another mwp
            if "Failed to find a valid instantiation" in str(e):
                continue
            else:
                raise e    
        
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

        # 6. foreach instantiation
        skip_mwp = False
        for inst_id,inst in enumerate(variations):
            random.seed(seed)
            seed = random.randint(0, 2**32 - 1)

            # reset all rules to the original/correct ones
            for n in mwp.tree.traverse():
                n.rule = orig_rules_by_node[n]

            # NOTE: cannot assume tree structure to be the original one in this loop (could have been changed in-place from the previous instantiation)
            mwp.update_instantiation(inst)

            # 6.a select new misconcievable conclusion (either if we want to re-select it for each instantiation or if its the first time)
            if vary_applicable or inst_id == 0: 
                # choose a couple rules that will be re-formulated s.t. misconceptions are applicable
                misconcievable_conclusions = []
                misconcievable_conclusion_ids = []
                try:
                    misconcievable_conclusions = choose_potentially_misconcievable_conclusions(mwp.tree, list(miscon_ruleset.keys()), prob_select=prob_misconcievable, seed=seed)
                    misconcievable_conclusion_ids = [mwp.tree.id_by_node[n] for n in misconcievable_conclusions]
                    
                    if len(misconcievable_conclusions) >= 6:
                        print(f"Warning: The number of misconcievable conclusions is quite high {len(misconcievable_conclusions)}. Beware that the complexity grows exponentially! Consider reducing the likelihood of selecting each rule to be phrased inconsistently.")
                except ValueError as e:
                    if "Need at least one conclusion where a misconception is applicable to choose a non-empty set of nodes!" in str(e) or "Failed to find a non-empty subset of conclusions which are misconcievable!" in str(e):
                        if omit_non_applicable:
                            skip_mwp = True # abort and then try the next mwp
                            break 
                        else:
                            pass # continue without any applicable nodes
                    else:
                        raise e

                # mark the premises of the misconcievable conclusions as inconsistent
                inconsistent_nodes = []
                for misconcievable_conclusion in misconcievable_conclusions:
                    misconcievable_premises = [c for c in misconcievable_conclusion.child_nodes if c.is_leaf]
                    assert len(misconcievable_premises) > 0, "Should always have at least one premise that is an axiom"
                    # TODO: and we should be able to flip this axiom...
                    inconsistent_nodes.extend(misconcievable_premises)
            
                # initial render of the problem using inconsistent language only where necessary
                mwp.problem_as_nl(seed=seed) # render consistent
                con_template_selections = mwp.ps_meta.template_selections
                
                mwp.problem_as_nl(override_sampler_by_node_id={mwp.tree.id_by_node[n]:inconsistent_sampler for n in inconsistent_nodes}, seed=seed)
                incon_template_selections = mwp.ps_meta.template_selections

            # 6.2. re-render the newly instantiated problem...
            if vary_formulations:
                # ... using potentially different formulations from the original rendering
                mwp.problem_as_nl(seed=seed)
                problem_con = mwp.ps_nl
                mwp.problem_as_nl(override_sampler_by_node_id={mwp.tree.id_by_node[n]:inconsistent_sampler for n in inconsistent_nodes}, seed=seed)
                incon_template_selections = mwp.ps_meta.template_selections
                problem_incon = mwp.ps_nl
            else:
                # ... using the same formulations as for the original rendering
                mwp.problem_as_nl(preselected_templates=con_template_selections)
                problem_con = mwp.ps_nl
                mwp.problem_as_nl(preselected_templates=incon_template_selections)
                problem_incon = mwp.ps_nl
            
            all_op_pert_erroneous_answers = None
            if include_op_pert:
                all_op_pert_erroneous_answers = compute_all_op_pert_erroneous_answers(orig_expression, mwp.instantiation)

            mwp_data["instantiations"][inst_id] = {
                "inconsistent_conclusions": misconcievable_conclusion_ids,
                "problem": problem_incon,
                "cons_problem": problem_con if include_cons_form else None,
                "instantiation": {str(k):v for k,v in mwp.instantiation.get_instantiations_of_type(PropertyType.QUANTITY).items()},
                "correct_answer": -1,
                "misconception_answers": [], # potential answers when only inconsistent phrasings are misunderstood
                "all_op_perturbation_answers": all_op_pert_erroneous_answers # all answers one might get by flipping any of the operators (may include the correct answer too if it's achievable via another combination)
            }

            # 6.2. iterate over all potential answers the model could come up with under this instantiation
            for v in iter_all_errors_rec(mwp, misconcievable_conclusions, orig_rules_by_node, FULL_MISCONCEPTIONS_BY_RULE_TYPE):
                if v is None: break

                # compute the quantities on all nodes of the tree
                all_results = [
                    quantity.eval(mwp.instantiation)
                    for node in v.tree.traverse_reasoning_trace(v.problem_order.body_node_ids)
                    for quantity in node.logicalform.get_quantities()
                ]

                # render the reasoning trace if asked
                if include_rt:
                    v.reasoning_trace_as_nl(enforce_premise_axiom_consistency=True)

                is_correct_answer = all(n.is_leaf or (n.rule == orig_rules_by_node[n]) for n in v.tree.nodes) 
                
                if is_correct_answer:
                    # correct answer
                    assert all(r >= 2 and r < 1000 and int(r) == r for r in all_results), "Correct answer should always be plausible!"
                    mwp_data["instantiations"][inst_id]["correct_answer"] = {
                        "answer": v.numerical_answers[0]
                    }

                    if include_rt:
                        mwp_data["instantiations"][inst_id]["correct_answer"]["rt"] = v.rt_nl
                else:
                    # error due to inconsistent formulation / misconception
                    inconsistent_option = {
                        "answer": v.numerical_answers[0],
                        "expression": str(v.tree.root_node.logicalform.get_quantities()[0]),
                        "misconceptions": {
                            v.tree.id_by_node[n]: type(n.rule).__name__
                            for n in v.tree.traverse_reasoning_trace(v.problem_order.body_node_ids)
                            if not n.is_leaf and (n in misconcievable_conclusions) and (n.rule != orig_rules_by_node[n])
                        },
                        "plausible": all(r >= 2 and r < 1000 and int(r) == r for r in all_results)
                    }

                    if include_rt:
                        inconsistent_option["rt"] = v.rt_nl

                    mwp_data["instantiations"][inst_id]["misconception_answers"].append(inconsistent_option)
                    
                    assert all_op_pert_erroneous_answers is None or v.numerical_answers[0] in all_op_pert_erroneous_answers, f"All misconception answers should also be part of the general op-pert set but {v.numerical_answers[0]} is not in {sorted(all_op_pert_erroneous_answers)}"

        if skip_mwp: # skip this mwp (e.g. due to a structure where no errors are applicable etc)
            continue
        
        dataset[mwp_id] = mwp_data    
    return dataset