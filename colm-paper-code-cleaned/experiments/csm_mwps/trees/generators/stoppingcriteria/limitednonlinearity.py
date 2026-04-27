
from typing import List
from mathgap.logicalforms.container import Container
from mathgap.logicalforms.logicalform import LogicalForm
from mathgap.trees.generators.policies.uniform import UniformPolicy
from mathgap.trees.generators.stoppingcriteria.criterion import Criterion
from mathgap.trees.generators.stoppingcriteria.depth import BranchDepthCriterion
from mathgap.trees.prooftree import ProofTree, TreeNode
from mathgap.trees.rules.comp.cont_cont import ContContComp
from mathgap.trees.rules.container.cont_comp_compeq import ContCompCompeqCont
from mathgap.trees.rules.inference_rule import InferenceRule

class AvoidContContCompOfCompeqExpansion(Criterion):
    """ Avoids the expansion of Containers that lead to a Comp of a Compeq"""
    def satisfied(self, node: TreeNode, tree: ProofTree) -> bool:
        if isinstance(node.logicalform, Container):
            parent = tree.parent_by_node.get(node, None)
            if parent is not None and type(parent.rule) == ContContComp:
                parent2 = tree.parent_by_node.get(parent, None)
                if parent2 is not None:
                    if type(parent2.rule) == ContCompCompeqCont:
                        return True
        return False