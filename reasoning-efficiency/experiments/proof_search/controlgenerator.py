from typing import Dict, List, Tuple, Generator as GenType
import random

from mathgap.trees.generators.general import GeneralGenerator
from mathgap.trees.generators.generator import Generator

from mathgap.trees.prooftree import ProofTree
from mathgap.trees.rules import InferenceRule, ContContComp, ContCompCompeqCont, ContTransferCont, ContCompCont, ContPartWhole, ContRateCont
from mathgap.logicalforms import LogicalForm, Container, ComparisonType, Comp, PartWhole
from mathgap.properties import PropertyType, PropertyTracker

class ControlGenerator(Generator):
    def __init__(self, base_tree: ProofTree, rule_stack: List[InferenceRule], base_generator: GeneralGenerator, desired_width: int) -> None:
        """ 
            Generates a tree of approximately the desired width by extending the base_tree with random rules from rule_stack whenever possible
            - base_tree: base_tree to start generating from
            - rule_stack: rules that should be inc
        """
        super().__init__(base_generator.start_types, base_generator.inference_rules, base_generator.stopping_criterion)
        self.base_tree = base_tree
        self.rule_stack = rule_stack
        self.base_generator = base_generator
        self.desired_width = desired_width

    def _request_var(self, property_tracker: PropertyTracker, use_entity: bool|int=True, use_attribute: bool|None|int=False, use_unit: bool|None|int=False):
        entity = None
        if isinstance(use_entity, bool):
            if use_entity == True:
                entity = property_tracker.request_id(PropertyType.ENTITY)
        elif isinstance(use_entity, int):
            entity = use_entity
        
        attribute = None
        if isinstance(use_attribute, bool):
            if use_attribute == True:
                attribute = property_tracker.request_id(PropertyType.ATTRIBUTE)
        elif isinstance(use_attribute, int):
            attribute = use_attribute

        unit = None
        if isinstance(use_unit, bool):
            if use_unit == True:
                unit = property_tracker.request_id(PropertyType.UNIT)
        elif isinstance(use_unit, int):
            unit = use_unit

        return (
            property_tracker.request_id(PropertyType.AGENT),
            entity,
            attribute,
            unit
        )

    def generate(self, seed: int = 14) -> ProofTree:
        random.seed(seed)
        rule_stack = self.rule_stack.copy()
        use_attribute, use_unit = self.base_generator.use_attribute, self.base_generator.use_unit

        def var_from_container(cont: Container):
            return (cont.agent, cont.entity, cont.attribute, cont.unit)
        
        def set_parametrization(parametrization: Dict, var_name: str, var):
            parametrization[f"{var_name}_agent"] = var[0] 
            parametrization[f"{var_name}_entity"] = var[1] 
            parametrization[f"{var_name}_attribute"] = var[2] 
            parametrization[f"{var_name}_unit"] = var[3] 

        tree = self.base_tree.copy()
        root = tree.root_node.logicalform
        property_tracker = tree.property_tracker

        part_whole_entities = set([])
        for lf, valid_stack_rules, valid_rules in self.expand_bfs(tree, rule_stack, root, property_tracker):
            if len(valid_stack_rules) > 0:
                # sample uniformly from the stack
                rule = random.choice(valid_stack_rules)
                rule_stack.remove(rule) # remove one instance of rule
            else:
                # fallback to default generator
                rule = self.base_generator.rule_sampling_policy.sample(lf, tree, valid_rules, self.stopping_criterion)
            parametrization = {}
            if isinstance(rule, ContTransferCont):
                assert isinstance(lf, Container), "Conclusion is expected to be Container"
                vars = [lf.agent]

                # if we start with a part-whole then no other agent should have any such entity
                # e.g. A has 4 apples. B has 3 peaches. C gets 3 peaches from B. How many peaches does everyone combined have? => 7 bc A has 4, B has 0 and C has 3
                if lf.entity not in part_whole_entities:
                    vars.append(property_tracker.request_id(PropertyType.AGENT))
                else:
                    vars.append(None)

                random.shuffle(vars)

                parametrization["sender_agent"] = vars[0]
                parametrization["receiver_agent"] = vars[1]
                parametrization["attribute"] = lf.attribute # TODO: allow introduction of attribute if none and no unit
                parametrization["unit"] = lf.unit
            elif isinstance(rule, ContCompCont):
                assert isinstance(lf, Container), "Conclusion is expected to be Container"
                comp_same_entity = random.random() <= self.base_generator.comp_same_entity_prob

                ue,ua,uu = True, use_attribute, use_unit
                if comp_same_entity:
                    ue,ua,uu = lf.entity, lf.attribute, lf.unit

                vars = [
                    var_from_container(lf),
                    self._request_var(property_tracker, use_entity=ue, use_attribute=ua, use_unit=uu),
                ]
                random.shuffle(vars)

                set_parametrization(parametrization, "subj", vars[0])
                set_parametrization(parametrization, "obj", vars[1])

                parametrization["comp_type"] = random.choice(self.base_generator.comp_allowed_comparisons)
            elif isinstance(rule, ContRateCont):
                assert isinstance(lf, Container), "Conclusion is expected to be Container"
                attribute = None
                if isinstance(use_attribute, bool):
                    if use_attribute == True:
                        attribute = property_tracker.request_id(PropertyType.ATTRIBUTE)
                elif isinstance(use_attribute, int):
                    attribute = use_attribute

                unit = None
                if isinstance(use_unit, bool):
                    if use_unit == True:
                        unit = property_tracker.request_id(PropertyType.UNIT)
                elif isinstance(use_unit, int):
                    unit = use_unit

                parametrization["super_entity"] = property_tracker.request_id(PropertyType.ENTITY)
                parametrization["super_attribute"] = attribute
                parametrization["super_unit"] = unit
            elif isinstance(rule, ContCompCompeqCont):
                assert isinstance(lf, Container), "Conclusion is expected to be Container"

                compeq_same_entity = random.random() <= self.base_generator.compeq_same_entity_prob

                ue,ua,uu = True, use_attribute, use_unit
                if compeq_same_entity:
                    ue,ua,uu = lf.entity, lf.attribute, lf.unit

                # NOTE: conclusion must match with subj or obj and not other_subj or other_obj 
                vars = [
                    var_from_container(lf),
                    self._request_var(property_tracker, use_entity=ue, use_attribute=ua, use_unit=uu),
                ]
                random.shuffle(vars)
                vars += [
                    self._request_var(property_tracker, use_entity=ue, use_attribute=ua, use_unit=uu),
                    self._request_var(property_tracker, use_entity=ue, use_attribute=ua, use_unit=uu),
                ]

                set_parametrization(parametrization, "subj", vars[0])
                set_parametrization(parametrization, "obj", vars[1])
                set_parametrization(parametrization, "other_subj", vars[2])
                set_parametrization(parametrization, "other_obj", vars[3])

                parametrization["comp_type"] = random.choice([ComparisonType.MORE_THAN]) # , ComparisonType.LESS_THAN
                parametrization["other_comp_type"] = random.choice([ComparisonType.MORE_THAN]) #, ComparisonType.LESS_THAN

            elif isinstance(rule, ContContComp):
                assert isinstance(lf, Comp), "Conclusion is expected to be a Comparison"
                # NOTE: no parametrization

            elif isinstance(rule, ContPartWhole):
                assert isinstance(lf, PartWhole), "Conclusion is expected to be a PartWhole"
                # NOTE: no parametrization

            premises = rule.apply_reverse(lf, parametrization)
            tree.add_derivation(premises, lf, rule)

            assert tree.validate(), "Should not be able to generate invalid trees!"

        tree.compute_symbolically()
        return tree
    
    def expand_bfs(self, tree: ProofTree, rule_stack: List[InferenceRule], root: LogicalForm, property_tracker: PropertyTracker) -> GenType[Tuple[LogicalForm, List[InferenceRule]], None, None]:
        """ Gradually tries to expand nodes in a BFS manner """
        leaves_queue = [node.logicalform for node in tree.leaf_nodes]
        while len(leaves_queue) > 0:
            leaf = leaves_queue.pop(0)
            if tree.width < self.desired_width:
                valid_stack_rules = [r for r in rule_stack if r.is_reverse_applicable(leaf, tree)]
                valid_rules = [r for r in self.inference_rules if r.is_reverse_applicable(leaf, tree)]
                # try expand the next node
                if len(valid_stack_rules) > 0 or len(valid_rules) > 0: # we can actually extend on this node
                    yield (leaf, valid_stack_rules, valid_rules)
                    premises = tree.nodes_by_lf[leaf].premises
                    leaves_queue.extend(premises)
                else:
                    # if we cannot extend => mark it as an axiom
                    leaf.make_axiom(property_tracker)
            else:
                # mark the final nodes as axioms
                leaf.make_axiom(property_tracker)