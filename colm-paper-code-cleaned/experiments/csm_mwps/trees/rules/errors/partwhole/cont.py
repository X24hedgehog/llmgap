from typing import List, Dict
from mathgap.trees.rules.inference_rule import InferenceRule, Parametrization

from mathgap.logicalforms import LogicalForm, Container, PartWhole
from mathgap.expressions import Sum
from mathgap.trees.rules.partwhole.cont import ContPartWhole

from datageneration.arithmeticerrors import MentalModel

class ContPartWholeArithError(ContPartWhole):
    """ Partwhole but student makes arithmetic errors according to mental_model (passed by ref, so it can be changed later) """
    def __init__(self, mental_model: MentalModel):
        self.mental_model = mental_model

    def infer_knowledge(self, premises: List[Container], conclusion: LogicalForm):
        assert all([isinstance(c, Container) for c in premises]), "All premises are expected to be a Container"
        assert isinstance(conclusion, PartWhole), "Conclusion is expected to be a PartWhole"
        assert all(c.quantity is not None for c in premises), "No containers quantity cannot be None"

        conclusion.quantity = Sum([c.quantity for c in premises])

        assert conclusion.quantity is not None, f"Conclusion.quantity must be set after inferring knowledge!"