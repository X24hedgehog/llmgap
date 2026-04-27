from typing import Dict, List
from mathgap.trees.rules import ContTransferCont

from mathgap.logicalforms import LogicalForm, Container, Transfer
from mathgap.expressions import Subtraction, Addition

from datageneration.arithmeticerrors import AdditionErr, MentalModel, SubtractionErr

class ContTransferContFlipOpError(ContTransferCont):
    """ ContTransferCont but operators are flipped (e.g. + becomes -) """
    def infer_knowledge(self, premises: List[LogicalForm], conclusion: LogicalForm):
        container, transfer = premises
        assert isinstance(container, Container), "First premise is expected to be a Container"
        assert isinstance(transfer, Transfer), "Second premise is expected to be a Transfer"
        assert isinstance(conclusion, Container), "Conclusion is expected to be a Container"
        assert container.agent == conclusion.agent, "Must operate on the same agent"
        assert container.entity == conclusion.entity, "Must operate on the same entity"
        assert container.attribute == conclusion.attribute, "Must operate on the same attribute"
        assert container.quantity is not None, "Container quantity cannot be None"
        assert transfer.quantity is not None, "Transfer quantity cannot be None"

        if conclusion.agent == transfer.receiver:
            conclusion.quantity = Subtraction(container.quantity, transfer.quantity)
        elif conclusion.agent == transfer.sender:
            conclusion.quantity = Addition(container.quantity, transfer.quantity)

        assert conclusion.quantity is not None, f"Conclusion.quantity must be set after inferring knowledge! conclusion.agent={conclusion.agent}, transfer.receiver={transfer.receiver}, transfer.sender={transfer.sender}"


class ContTransferContArithError(ContTransferCont):
    """ ContTransferCont but the student makes arithmetic errors according to a mental model (passed by ref, so it can be changed later)"""
    def __init__(self, mental_model: MentalModel):
        self.mental_model = mental_model
        
    def infer_knowledge(self, premises: List[LogicalForm], conclusion: LogicalForm):
        container, transfer = premises
        assert isinstance(container, Container), "First premise is expected to be a Container"
        assert isinstance(transfer, Transfer), "Second premise is expected to be a Transfer"
        assert isinstance(conclusion, Container), "Conclusion is expected to be a Container"
        assert container.agent == conclusion.agent, "Must operate on the same agent"
        assert container.entity == conclusion.entity, "Must operate on the same entity"
        assert container.attribute == conclusion.attribute, "Must operate on the same attribute"
        assert container.quantity is not None, "Container quantity cannot be None"
        assert transfer.quantity is not None, "Transfer quantity cannot be None"

        if conclusion.agent == transfer.receiver:
            conclusion.quantity = AdditionErr(container.quantity, transfer.quantity, self.mental_model)
        elif conclusion.agent == transfer.sender:
            conclusion.quantity = SubtractionErr(container.quantity, transfer.quantity, self.mental_model)

        assert conclusion.quantity is not None, f"Conclusion.quantity must be set after inferring knowledge! conclusion.agent={conclusion.agent}, transfer.receiver={transfer.receiver}, transfer.sender={transfer.sender}"