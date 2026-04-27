from typing import Dict, List

from mathgap.trees.generators.policies.rulesamplingpolicy import RuleSamplingPolicy
from mathgap.trees.prooftree import ProofTree

from mathgap.logicalforms import LogicalForm
from mathgap.trees.rules import InferenceRule

class NonlinearMoreLikelyWithDepthPolicy(RuleSamplingPolicy):
    """ 
        Policy that makes sampling nonlinear rules more likely at greater depths 
    
        Will have probability 0 until start_depth and then linearly increase to uniform probability over all rules by uniform_depth
    """
    def __init__(self, start_depth: int, uniform_depth: int, non_linear_rules: List[InferenceRule]):
        self.start_depth = start_depth
        self.uniform_depth = uniform_depth
        self.non_linear_rules = set(non_linear_rules)

    def get_probs(self, lf: LogicalForm, tree: ProofTree, rules: List[InferenceRule]) -> Dict[InferenceRule, float]:
        depth = tree.nodes_by_lf[lf].depth

        rel_weights = {}
        for rule in rules:
            if rule in self.non_linear_rules:
                rel_weights[rule] = max(min((depth - self.start_depth) / (self.uniform_depth - self.start_depth), 1.0), 0.0)
            else:
                rel_weights[rule] = 1.0

        # normalize the weights to sum up to one
        total_weights = sum(rel_weights.values())
        probs = {r:(w / total_weights) for r,w in rel_weights.items()} 
        
        return probs 