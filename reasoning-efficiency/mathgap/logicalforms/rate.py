from typing import Dict, List
from enum import Enum
from mathgap.properties import PropertyTracker
from mathgap.logicalforms.logicalform import LogicalForm, EntitySpec

from mathgap.properties import PropertyKey, PropertyType
from mathgap.expressions import Expr, Variable
from mathgap.trees.timing import VariableKey

class Rate(LogicalForm):
    """ 
        Puts in relation the quantity of the subject with the quantity of the subject
        I.e. Subj. has 5 more apples than obj has peaches.
    """
    def __init__(self, agent: int, quantity: Expr, 
                 super_entity: int, super_attribute: int, super_unit: int, 
                 sub_entity: int, sub_attribute: int, sub_unit: int) -> None:
        self.agent = agent
        self.quantity = quantity
        self.super_entity = super_entity
        self.super_attribute = super_attribute
        self.super_unit = super_unit
        self.sub_entity = sub_entity
        self.sub_attribute = sub_attribute
        self.sub_unit = sub_unit

    @property
    def agent_prop(self) -> PropertyKey:
        return PropertyKey(PropertyType.AGENT, self.agent)

    @property
    def quantity_prop(self) -> PropertyKey:
        return PropertyKey(PropertyType.QUANTITY, self.quantity)

    @property
    def super_entity_prop(self) -> PropertyKey:
        return PropertyKey(PropertyType.ENTITY, self.super_entity)

    @property
    def super_attribute_prop(self) -> PropertyKey:
        return PropertyKey(PropertyType.ATTRIBUTE, self.super_attribute)

    @property
    def super_unit_prop(self) -> PropertyKey:
        return PropertyKey(PropertyType.UNIT, self.super_unit)
    
    @property
    def sub_entity_prop(self) -> PropertyKey:
        return PropertyKey(PropertyType.ENTITY, self.sub_entity)

    @property
    def sub_attribute_prop(self) -> PropertyKey:
        return PropertyKey(PropertyType.ATTRIBUTE, self.sub_attribute)

    @property
    def sub_unit_prop(self) -> PropertyKey:
        return PropertyKey(PropertyType.UNIT, self.sub_unit)
    
    def get_available_properties(self) -> Dict[str, PropertyKey|List[PropertyKey]]:
        available_properties = {}
        if self.agent is not None: available_properties["agent"] = self.agent_prop
        if self.quantity is not None: available_properties["quantity"] = self.quantity_prop
        if self.super_entity is not None: available_properties["super_entity"] = self.super_entity_prop
        if self.super_attribute is not None: available_properties["super_attribute"] = self.super_attribute_prop
        if self.super_unit is not None: available_properties["super_unit"] = self.super_unit_prop
        if self.sub_entity is not None: available_properties["sub_entity"] = self.sub_entity_prop
        if self.sub_attribute is not None: available_properties["sub_attribute"] = self.sub_attribute_prop
        if self.sub_unit is not None: available_properties["sub_unit"] = self.sub_unit_prop
        return available_properties
    
    def get_entity_specs(self) -> List[EntitySpec]:
        return [
            EntitySpec(self.super_entity, attribute_id=self.super_attribute, unit_id=self.super_unit),
            EntitySpec(self.sub_entity, attribute_id=self.sub_attribute, unit_id=self.sub_unit, super_entity_id=self.super_entity)
        ]
    
    def get_variable_keys(self) -> List[VariableKey]:
        return [
            VariableKey((self.agent_prop, self.super_entity_prop, self.super_attribute_prop, self.super_unit_prop)),
            VariableKey((self.agent_prop, self.sub_entity_prop, self.sub_attribute_prop, self.sub_unit_prop))
        ]
    
    def get_quantities(self) -> List[Expr]:
        return [self.quantity]

    def make_axiom(self, property_tracker: PropertyTracker) -> None:
        self.quantity = Variable(property_tracker.request_key(PropertyType.QUANTITY))