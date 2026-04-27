# CONCLUSION: Container
# PREMISES: Container, Rate

# VARIABLE-TIMES: 
# - container(a.super at t1), rate(a.super at {t1}, a.sub at {t1})
# - conclusion(a.sub at t1)

# INPUTS:
# - both of them, with one of them matching the conclusion (attributes and units can ofc be None):
#   - super_entity, super_attribute, super_unit
#   - sub_entity, sub_attribute, sub_unit
# - agent

from typing import List
from mathgap.trees.rules.inference_rule import InferenceRule, Parametrization

from mathgap.logicalforms import LogicalForm, Container, Rate
from mathgap.expressions import Product

class ContRateCont(InferenceRule):
    """ 
        Informal:
            We know how many super-entities agent has, and we know how many sub-entities agent has per super-entity, 
            thus we know how many sub-entities agent has.
    """
    def is_reverse_applicable(self, conclusion: LogicalForm, tree) -> bool:
        if isinstance(conclusion, Container): return True
        # TODO: ensure theres no rate between conclusion and the tree root already
        return False
    
    def assert_valid_parametrization(self, conclusion: LogicalForm, parametrization: Parametrization):
        assert isinstance(conclusion, Container), f"Conclusion is expected to be a Container! Got {type(conclusion)} instead."
        assert "super_entity" in parametrization and "super_attribute" in parametrization and "super_unit" in parametrization,\
            f"Requires a parametrization of super_entity, super_attribute, super_unit"

        super = (parametrization["super_entity"], parametrization["super_attribute"], parametrization["super_unit"])
        concl = (conclusion.entity, conclusion.attribute, conclusion.unit)

        assert super != concl, f"Super must be different from conclusion"

        ignored_keys = set(parametrization.keys()).difference(set(["agent", "super_entity", "super_attribute", "super_unit", "sub_entity", "sub_attribute", "sub_unit"]))
        assert len(ignored_keys) == 0, f"Keys {ignored_keys} are provided but are not allowed."


    def apply_reverse(self, conclusion: LogicalForm, parametrization: Parametrization) -> List[LogicalForm]:
        self.assert_valid_parametrization(conclusion, parametrization)
        assert isinstance(conclusion, Container), "Conclusion is expected to be a Container"

        super_entity = parametrization["super_entity"]
        super_attribute = parametrization["super_attribute"]
        super_unit = parametrization["super_unit"]

        prem_container = Container(agent=conclusion.agent, quantity=None, entity=super_entity, attribute=super_attribute, unit=super_unit)
        prem_comp = Rate(agent=conclusion.agent, quantity=None,
                         super_entity=super_entity, super_attribute=super_attribute, super_unit=super_unit,
                         sub_entity=conclusion.entity, sub_attribute=conclusion.attribute, sub_unit=conclusion.unit)
        
        # TODO: could flip the order (all possible orders are deducible by the time-dag)
        return [prem_container, prem_comp]
        
    def infer_knowledge(self, premises: List[LogicalForm], conclusion: LogicalForm):
        container, rate = premises
        assert isinstance(container, Container), "First premise is expected to be a Container"
        assert isinstance(rate, Rate), "Second premise is expected to be a Rate"
        assert isinstance(conclusion, Container), "Conclusion is expected to be a Container"
        assert container.quantity is not None, "Container quantity cannot be None"
        assert rate.quantity is not None, "Comparison quantity cannot be None"

        conclusion.quantity = Product(container.quantity, rate.quantity)
        assert conclusion.quantity is not None, f"Conclusion.quantity must be set after inferring knowledge! comp.subj_agent={comp.subj_agent}, comp.obj_agent={comp.obj_agent}, conclusion.agent={conclusion.agent}, comp.comp_type={comp.comp_type}" 