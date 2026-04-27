from typing import Dict, List, Set

from mathgap.logicalforms import LogicalForm
from mathgap.trees.prooftree import ProofTree, TreeNode

class ProblemOrder:
    def __init__(self, body_node_ids: List[int], question_node_ids: List[int], body_render: List[bool]|None = None, question_render: List[bool]|None = None):
        """ 
            Basically an uninstantiated world model representing a math word problem on a prooftree
            (i.e. speficies in which nodes will constitute the body and question of the MWP and in what order)

            - body_node_ids: the order in which tree nodes form the body of the mwp
            - question_node_ids: the order in which tree nodes form the question of the mwp
            - body_render: which body nodes should be rendered
            - question_render: which question nodes should be rendered
        """
        self.body_node_ids = body_node_ids
        self.question_node_ids = question_node_ids
        
        if body_render is None:
            self.body_render = [True]*len(body_node_ids)
        else:
            self.body_render = body_render
        
        if question_render is None:
            self.question_render = [True]*len(question_node_ids)
        else:
            self.question_render = question_render

    def get_body(self, tree: ProofTree) -> List[LogicalForm]:
        # NOTE: should use the same tree structure that was also used to generate the ids in the first place
        return [tree.node_by_id[i].logicalform for i in self.body_node_ids]

    def get_questions(self, tree: ProofTree) -> List[LogicalForm]:
        # NOTE: should use the same tree structure that was also used to generate the ids in the first place
        return [tree.node_by_id[i].logicalform for i in self.question_node_ids]
    
    def hide_body_ids(self, ids: Set[int]) -> 'ProblemOrder':
        self.body_render = [i not in ids for i in self.body_node_ids]
        return self
    
    def show_only_body_ids(self, ids: Set[int]) -> 'ProblemOrder':
        self.body_render = [i in ids for i in self.body_node_ids]
        return self

    def hide_questions(self) -> 'ProblemOrder':
        self.question_render = [False]*len(self.question_node_ids)
        return self
    
    def is_rendered(self, node_id: int) -> bool:
        for b,r in zip(self.body_node_ids, self.body_render):
            if b == node_id and r:
                return True
        for q,r in zip(self.question_node_ids, self.question_render):
            if q == node_id and r:
                return True
        return False
    
    def __repr__(self):
        from mathgap.renderers import TEXT_RENDERER
        return TEXT_RENDERER(self)