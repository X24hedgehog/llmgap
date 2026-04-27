from typing import Dict, List
from mathgap.trees.rules import ContCompCont

from mathgap.logicalforms import LogicalForm, Container, Comp, ComparisonType
from mathgap.expressions import Addition, Fraction, Product, Subtraction

from datageneration.arithmeticerrors import AdditionErr, MentalModel, SubtractionErr

class ContCompContFlipOpError(ContCompCont):
      """ ContCompCont but operators are flipped (e.g. + becomes -) """
      def infer_knowledge(self, premises: List[LogicalForm], conclusion: LogicalForm):
        container, comp = premises
        assert isinstance(container, Container), "First premise is expected to be a Container"
        assert isinstance(comp, Comp), "Second premise is expected to be a Comparison"
        assert isinstance(conclusion, Container), "Conclusion is expected to be a Container"
        assert container.quantity is not None, "Container quantity cannot be None"
        assert comp.quantity is not None, "Comparison quantity cannot be None"

        if comp.obj_agent == conclusion.agent:
            # subj has + subj compared to obj => obj has 
            assert comp.subj_agent == container.agent and comp.subj_entity == container.entity and comp.subj_attribute == container.attribute,\
                "Expects the container to be about the subj of the comparison"
            assert comp.obj_agent == conclusion.agent and comp.obj_entity == conclusion.entity and comp.obj_attribute == conclusion.attribute,\
                "Expects the conclusion to be about the obj of the comparison"
            
            if comp.comp_type == ComparisonType.MORE_THAN:
                conclusion.quantity = Addition(container.quantity, comp.quantity)
            elif comp.comp_type == ComparisonType.LESS_THAN:
                conclusion.quantity = Subtraction(container.quantity, comp.quantity)
            elif comp.comp_type == ComparisonType.TIMES_AS_MANY:
                conclusion.quantity = Product(container.quantity, comp.quantity)
            elif comp.comp_type == ComparisonType.FRACTION_OF:
                conclusion.quantity = Fraction(container.quantity, comp.quantity)

        elif comp.subj_agent == conclusion.agent:
            # obj has + subj compared to obj => subj has
            assert comp.obj_agent == container.agent and comp.obj_entity == container.entity and comp.obj_attribute == container.attribute,\
                "Expects the container to be about the obj of the comparison"
            assert comp.subj_agent == conclusion.agent and comp.subj_entity == conclusion.entity and comp.subj_attribute == conclusion.attribute,\
                "Expects the conclusion to be about the subj of the comparison"
            
            if comp.comp_type == ComparisonType.MORE_THAN:
                conclusion.quantity = Subtraction(container.quantity, comp.quantity)
            elif comp.comp_type == ComparisonType.LESS_THAN:
                conclusion.quantity = Addition(container.quantity, comp.quantity)
            elif comp.comp_type == ComparisonType.TIMES_AS_MANY:
                conclusion.quantity = Fraction(container.quantity, comp.quantity)
            elif comp.comp_type == ComparisonType.FRACTION_OF:
                conclusion.quantity = Product(container.quantity, comp.quantity)

        assert conclusion.quantity is not None, f"Conclusion.quantity must be set after inferring knowledge! comp.subj_agent={comp.subj_agent}, comp.obj_agent={comp.obj_agent}, conclusion.agent={conclusion.agent}, comp.comp_type={comp.comp_type}" 

class ContCompContArithError(ContCompCont):
    """ ContCompCont but student makes arithmetic errors according to mental_model (passed by ref, so it can be changed later) """
    def __init__(self, mental_model: MentalModel):
        self.mental_model = mental_model

    def infer_knowledge(self, premises: List[LogicalForm], conclusion: LogicalForm):
        container, comp = premises
        assert isinstance(container, Container), "First premise is expected to be a Container"
        assert isinstance(comp, Comp), "Second premise is expected to be a Comparison"
        assert isinstance(conclusion, Container), "Conclusion is expected to be a Container"
        assert container.quantity is not None, "Container quantity cannot be None"
        assert comp.quantity is not None, "Comparison quantity cannot be None"

        if comp.obj_agent == conclusion.agent:
            # subj has + subj compared to obj => obj has 
            assert comp.subj_agent == container.agent and comp.subj_entity == container.entity and comp.subj_attribute == container.attribute,\
                "Expects the container to be about the subj of the comparison"
            assert comp.obj_agent == conclusion.agent and comp.obj_entity == conclusion.entity and comp.obj_attribute == conclusion.attribute,\
                "Expects the conclusion to be about the obj of the comparison"
            
            if comp.comp_type == ComparisonType.MORE_THAN:
                conclusion.quantity = SubtractionErr(container.quantity, comp.quantity, self.mental_model)
            elif comp.comp_type == ComparisonType.LESS_THAN:
                conclusion.quantity = AdditionErr(container.quantity, comp.quantity, self.mental_model)

        elif comp.subj_agent == conclusion.agent:
            # obj has + subj compared to obj => subj has
            assert comp.obj_agent == container.agent and comp.obj_entity == container.entity and comp.obj_attribute == container.attribute,\
                "Expects the container to be about the obj of the comparison"
            assert comp.subj_agent == conclusion.agent and comp.subj_entity == conclusion.entity and comp.subj_attribute == conclusion.attribute,\
                "Expects the conclusion to be about the subj of the comparison"
            
            if comp.comp_type == ComparisonType.MORE_THAN:
                conclusion.quantity = AdditionErr(container.quantity, comp.quantity, self.mental_model)
            elif comp.comp_type == ComparisonType.LESS_THAN:
                conclusion.quantity = SubtractionErr(container.quantity, comp.quantity, self.mental_model)

        assert conclusion.quantity is not None, f"Conclusion.quantity must be set after inferring knowledge! comp.subj_agent={comp.subj_agent}, comp.obj_agent={comp.obj_agent}, conclusion.agent={conclusion.agent}, comp.comp_type={comp.comp_type}" 