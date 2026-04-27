import random
from mathgap.trees.prooftree import ProofTree, TreeNode
from mathgap.trees.generators.stoppingcriteria.criterion import Criterion

class ProbabilisticBranchDepthCriterion(Criterion):
    """ 
        Limits the depth of each extendable tree-branch by making already deep branches less likely to be expanded
        
        Probability drops exponentially starting with 1.0 at depth min_depth
    """
    def __init__(self, min_depth, base = 2.0) -> None:
        self.min_depth = min_depth
        self.base = base

    def satisfied(self, node: TreeNode, tree: ProofTree) -> bool:
        return random.random() > min((self.base ** (- (node.depth - self.min_depth))), 1.0)