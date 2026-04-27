from typing import Dict, List, Optional
import os
import pickle
import re
from copy import deepcopy

from pydantic import BaseModel, Field
from mathgap.natlang.templates.sampling import TemplateSampler, TemplateSelection
from mathgap.trees import ProofTree
from mathgap.instantiate import Instantiation
from mathgap.logicalforms import Container, Transfer
from mathgap.natlang.templates import ReasoningTraceRenderer, ReasoningTraceSampler, ProblemStructureSampler, ProblemStructureRenderer, ProblemStructureAnswersSampler, RenderingMetadata, render_answers, render_problem, render_reasoning_trace
from mathgap.problemsample import ProblemOrder
from mathgap.trees.sampling import OrderSampler


class SubquestionTreeNode(BaseModel):
    node_id: int
    question: Optional[str] = None
    children: List["SubquestionTreeNode"] = Field(default_factory=list)


SubquestionTreeNode.model_rebuild()

class MathWordProblem(BaseModel):
    ps_nl: Optional[str] = Field(default=None) # problem with questions only
    answers_nl: Optional[str] = Field(default=None) # answers only
    rt_nl: Optional[str] = Field(default=None) # reasoning trace only
    subquestions_nl: Optional[List[str]] = Field(default=None) # ordered bottom-to-top
    subquestions_by_node_id: Optional[Dict[int, str]] = Field(default=None)
    subquestion_tree: Optional[SubquestionTreeNode] = Field(default=None)
    ps_meta: Optional[RenderingMetadata] = Field(default=None, exclude=True)
    answers_meta: Optional[RenderingMetadata] = Field(default=None, exclude=True)
    rt_meta: Optional[RenderingMetadata] = Field(default=None, exclude=True)
    
    tree: Optional[ProofTree] = Field(default=None, exclude=True)
    instantiation: Optional[Instantiation] = Field(default=None, exclude=True)
    
    ps_template_sampler: Optional[ProblemStructureSampler] = Field(default=None, exclude=True)
    answers_template_sampler: Optional[ProblemStructureAnswersSampler] = Field(default=None, exclude=True)
    ps_renderer: Optional[ProblemStructureRenderer] = Field(default=None, exclude=True)
    rt_template_sampler: Optional[ReasoningTraceSampler] = Field(default=None, exclude=True)
    rt_renderer: Optional[ReasoningTraceRenderer] = Field(default=None, exclude=True)

    problem_order: Optional[ProblemOrder] = Field(default=None, exclude=True)
    numerical_answers: Optional[List[int]] = Field(default=None) # numerical answers to each subquestion

    @staticmethod
    def _extract_last_question(text: str) -> str:
        text = " ".join(text.split())

        matches = re.findall(r'[^.?!]*\?', text)
        if len(matches) > 0:
            return matches[-1].strip()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 0:
            return lines[-1]

        return text.strip()

    def _render_entity_phrase(self, container: Container) -> str:
        assert self.instantiation is not None, "Need an instantiation first!"

        entity_value = self.instantiation[container.entity_prop]
        entity_text = entity_value[1] if isinstance(entity_value, tuple) and len(entity_value) > 1 else str(entity_value)

        if container.attribute is not None and container.attribute_prop in self.instantiation:
            attribute_text = str(self.instantiation[container.attribute_prop])
            return f"{attribute_text} {entity_text}"

        return entity_text

    def _render_transfer_aware_subquestion(self, node_id: int, default_question: str) -> str:
        assert self.tree is not None, "Need a tree first!"
        assert self.instantiation is not None, "Need an instantiation first!"

        node = self.tree.node_by_id[node_id]
        if node.is_leaf or type(node.rule).__name__ != "ContTransferCont":
            return default_question

        conclusion = node.logicalform
        if not isinstance(conclusion, Container):
            return default_question

        transfer_premise = next((premise for premise in node.premises if isinstance(premise, Transfer)), None)
        if transfer_premise is None:
            return default_question

        agent_text = str(self.instantiation[conclusion.agent_prop])
        entity_text = self._render_entity_phrase(conclusion)
        quantity_text = str(transfer_premise.quantity.eval(self.instantiation))
        receiver_text = str(self.instantiation[transfer_premise.receiver_prop])
        sender_text = str(self.instantiation[transfer_premise.sender_prop])

        if conclusion.agent == transfer_premise.sender:
            return f"How many {entity_text} does {agent_text} have after giving {receiver_text} {quantity_text} {entity_text}?"
        if conclusion.agent == transfer_premise.receiver:
            return f"How many {entity_text} does {agent_text} have after receiving {quantity_text} {entity_text} from {sender_text}?"

        return default_question

    def _render_subproblem_for_node(self, node_id: int, seed: int = 14) -> str:
        assert self.problem_order is not None, "Need to sample a problem structure first!"
        assert self.tree is not None, "Need a tree first!"
        assert self.instantiation is not None, "Need an instantiation first!"

        tmp_problem_order = deepcopy(self.problem_order)
        tmp_problem_order.question_node_ids = [node_id]
        tmp_problem_order.question_render = [True]

        target_node = self.tree.node_by_id[node_id]
        required_body_node_ids = {
            self.tree.id_by_node[leaf_node]
            for leaf_node in target_node.get_leaves()
        }
        tmp_problem_order.show_only_body_ids(required_body_node_ids)

        preselected_templates = None
        if self.ps_meta is not None:
            preselected_templates = [
                ts for ts in self.ps_meta.template_selections
                if self.tree.node_by_id[ts.primary_node_id].is_leaf
                and ts.primary_node_id in required_body_node_ids
            ]

        subproblem_nl, _ = render_problem(
            self.tree,
            self.instantiation,
            tmp_problem_order,
            self.ps_template_sampler,
            self.ps_renderer,
            preselected_templates=preselected_templates,
            seed=seed,
        )

        question = self._extract_last_question(subproblem_nl)
        transfer_aware_question = self._render_transfer_aware_subquestion(node_id=node_id, default_question=question)
        if transfer_aware_question != question:
            subproblem_nl = subproblem_nl.rstrip()
            if subproblem_nl.endswith(question):
                subproblem_nl = f"{subproblem_nl[:-len(question)]}{transfer_aware_question}"
            else:
                subproblem_nl = subproblem_nl.replace(question, transfer_aware_question)

        return subproblem_nl

    def _render_subquestion_for_node(self, node_id: int, seed: int = 14) -> str:
        subproblem_nl = self._render_subproblem_for_node(node_id=node_id, seed=seed)
        return self._extract_last_question(subproblem_nl)

    def build_subquestion_tree(self, seed: int = 14) -> SubquestionTreeNode:
        assert self.tree is not None, "Need a tree first!"
        assert self.problem_order is not None, "Need to sample a problem structure first!"
        assert self.instantiation is not None, "Need an instantiation first!"

        subquestions_by_node_id: Dict[int, str] = {}

        def _resolve_tree_node(node_or_lf):
            if node_or_lf in self.tree.id_by_node:
                return node_or_lf
            if node_or_lf in self.tree.nodes_by_lf:
                return self.tree.nodes_by_lf[node_or_lf]
            raise KeyError(f"Could not resolve proof tree node for object: {node_or_lf}")

        def _build(node_or_lf) -> SubquestionTreeNode:
            node = _resolve_tree_node(node_or_lf)
            node_id = self.tree.id_by_node[node]
            question = None

            if (not node.is_leaf) and (node != self.tree.root_node):
                question = self._render_subquestion_for_node(node_id=node_id, seed=seed + node_id)
                subquestions_by_node_id[node_id] = question

            children = []
            if not node.is_leaf:
                children = [_build(child) for child in node.premises]

            return SubquestionTreeNode(
                node_id=node_id,
                question=question,
                children=children
            )

        self.subquestion_tree = _build(self.tree.root_node)
        self.subquestions_by_node_id = subquestions_by_node_id
        return self.subquestion_tree

    def extract_subquestions(self) -> List[str]:
        assert self.tree is not None, "Need a tree first!"

        if self.subquestion_tree is None or self.subquestions_by_node_id is None:
            self.build_subquestion_tree()

        eligible_nodes = [
            node for node in self.tree.node_by_id.values()
            if (not node.is_leaf) and (node != self.tree.root_node)
        ]
        eligible_nodes = sorted(
            eligible_nodes,
            key=lambda node: (-node.depth, self.tree.id_by_node[node])
        )

        self.subquestions_nl = [
            self.subquestions_by_node_id[self.tree.id_by_node[node]]
            for node in eligible_nodes
        ]
        return self.subquestions_nl

    def subquestions_as_nl(self, seed: int = 14) -> List[str]:
        self.build_subquestion_tree(seed=seed)
        return self.extract_subquestions()
    

    def sample_problem_order(self, order_sampler: OrderSampler, seed: int = 14) -> ProblemOrder:
        """ Samples the problem, returns the visitation order """
        self.problem_order = order_sampler.sample_order(self.tree, seed)    

    def problem_as_nl(self, override_sampler_by_node_id: Dict[int, TemplateSampler] = None, preselected_templates: List[TemplateSelection] = None, seed: int = 14):
        """ Renders the problem structure as natural language """
        assert self.problem_order is not None, "Need to sample a problem structure first!"

        self.ps_nl, self.ps_meta = render_problem(self.tree, self.instantiation, self.problem_order, self.ps_template_sampler, self.ps_renderer, 
                                                  preselected_templates=preselected_templates, override_sampler_by_node_id=override_sampler_by_node_id, seed=seed)
        
    def reasoning_trace_as_nl(self, preselected_templates: List[TemplateSelection] = None, 
                              enforce_premise_axiom_consistency: bool = True, enforce_same_axiom_order: bool = True, seed: int = 14):
        """ Renders the reasoning trace for the problem structure as natural language """
        assert self.problem_order is not None, "Need to sample a problem structure first!"
        
        if self.ps_meta is not None:
            if preselected_templates is None and enforce_premise_axiom_consistency:
                # use the same way of expressing axioms as in the problem formulation
                preselected_templates = [ts for ts in self.ps_meta.template_selections if self.tree.node_by_id[ts.primary_node_id].is_leaf]
            else:
                raise NotImplementedError("Having both preselected templates while enforcing premise-axiom-consistency is not supported!")

        self.rt_nl, self.rt_meta = render_reasoning_trace(self.tree, self.instantiation, self.problem_order, self.rt_template_sampler, self.rt_renderer, 
                                                          preselected_templates=preselected_templates, enforce_premise_axiom_consistency=enforce_premise_axiom_consistency,
                                                          enforce_same_axiom_order=enforce_same_axiom_order, seed=seed)

    def answers_as_nl(self, preselected_templates: List[TemplateSelection] = None, seed: int = 14):
        """ Renders the answer(s) to the problem structure as natural language """
        assert self.problem_order is not None, "Need to sample a problem structure first!"

        self.answers_nl, self.answers_meta = render_answers(self.tree, self.instantiation, self.problem_order, self.answers_template_sampler, self.ps_renderer, 
                                                            preselected_templates=preselected_templates, seed=seed)

    def compute_answers(self):
        """ Compute the numerical answer(s) based on the instantiation and problem structure """
        assert self.problem_order is not None, "Requires problem structure to have been sampled"

        question_lfs = [self.tree.node_by_id[i].logicalform for i in self.problem_order.question_node_ids]
        self.numerical_answers = self.tree.instantiated_quantities(question_lfs, self.instantiation)
    
    def update_instantiation(self, instantiation: Instantiation):
        self.instantiation = instantiation

        self.ps_nl = None
        self.answers_nl = None
        self.rt_nl = None
        self.subquestions_nl = None
        self.subquestions_by_node_id = None
        self.subquestion_tree = None
        self.ps_meta = None
        self.answers_meta = None
        self.rt_meta = None
        self.numerical_answers = None

    def save(self, folder: str, name: str, include_metadata: bool = True, include_trees: bool = True, include_instantiation: bool = True):
        os.makedirs(folder, exist_ok=True)

        self_file = os.path.join(folder, f"{name}.json")
        with open(self_file, "w") as f:
            f.write(self.model_dump_json())

        if include_metadata:
            if self.ps_meta is not None:
                ps_meta_file = os.path.join(folder, f"{name}_ps_meta.pkz")
                with open(ps_meta_file, "wb") as f:
                    pickle.dump(self.ps_meta, f)

            if self.answers_meta is not None:
                answers_meta_file = os.path.join(folder, f"{name}_answers_meta.pkz")
                with open(answers_meta_file, "wb") as f:
                    pickle.dump(self.answers_meta, f)

            if self.rt_meta is not None:
                rt_meta_file = os.path.join(folder, f"{name}_rt_meta.pkz")
                with open(rt_meta_file, "wb") as f:
                    pickle.dump(self.rt_meta, f)

        if include_trees:
            tree_file = os.path.join(folder, f"{name}_tree.pkz")
            with open(tree_file, "wb") as f:
                pickle.dump(self.tree, f)

        if include_instantiation:
            instantiation_file = os.path.join(folder, f"{name}_instantiation.pkz")
            with open(instantiation_file, "wb") as f:
                pickle.dump(self.instantiation, f)

    def load(folder: str, name: str, sub_cls: type = None) -> 'MathWordProblem':
        self_file = os.path.join(folder, f"{name}.json")
        with open(self_file, "r") as f:
            sub_cls = MathWordProblem if sub_cls is None else sub_cls
            assert issubclass(sub_cls, MathWordProblem), "sub_cls needs to be either MathWordProblem or subtype thereof"
            model = sub_cls.model_validate_json(f.read())

        ps_meta_file = os.path.join(folder, f"{name}_ps_meta.pkz")
        if os.path.exists(ps_meta_file):
            with open(ps_meta_file, "rb") as f:
                model.ps_meta = pickle.load(f)

        answers_meta_file = os.path.join(folder, f"{name}_answers_meta.pkz")
        if os.path.exists(answers_meta_file):
            with open(answers_meta_file, "rb") as f:
                model.answers_meta = pickle.load(f)

        rt_meta_file = os.path.join(folder, f"{name}_rt_meta.pkz")
        if os.path.exists(rt_meta_file):
            with open(rt_meta_file, "rb") as f:
                model.rt_meta = pickle.load(f)

        tree_file = os.path.join(folder, f"{name}_tree.pkz")
        if os.path.exists(tree_file):
            with open(tree_file, "rb") as f:
                model.tree = pickle.load(f)

        instantiation_file = os.path.join(folder, f"{name}_instantiation.pkz")
        if os.path.exists(instantiation_file):
            with open(instantiation_file, "rb") as f:
                model.instantiation = pickle.load(f)

        return model

    class Config:
        arbitrary_types_allowed = True