from typing import Dict, List

from mathgap.trees.generators.policies.rulesamplingpolicy import RuleSamplingPolicy
from mathgap.trees.prooftree import ProofTree

from mathgap.logicalforms import LogicalForm
from mathgap.trees.rules import InferenceRule

class UniformPolicy(RuleSamplingPolicy):
    def get_probs(self, lf: LogicalForm, tree: ProofTree, rules: List[InferenceRule]) -> Dict[InferenceRule, float]:
        return {r: 1.0 / len(rules) for r in rules}