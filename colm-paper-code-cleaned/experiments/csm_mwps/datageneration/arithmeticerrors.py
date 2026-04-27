from typing import Any, Dict, List
from itertools import zip_longest
from mathgap.expressions import Expr
from pydantic import BaseModel, Field

class MentalModel(BaseModel):
    addition: str|None = Field(default=None)
    subtraction: str|None = Field(default=None)

    def set_correct(self):
        self.addition = None
        self.subtraction = None

class SumErr(Expr):
    summands: List[Expr]
    mental_model: MentalModel

    def __init__(self, summands: List[Expr], mental_model: MentalModel) -> None:
        super().__init__(subexpressions=summands, summands=summands, mental_model=mental_model)

    def _eval(self, instantiation):
        svs = [s.eval(instantiation) for s in self.summands]
        err = self.mental_model.addition
        
        if err is None or all(sv < 10 for sv in svs):
            # we require at least one 2-digit number for this to work
            return sum(svs)
        elif err == "regrouping":
            res = ""
            for cs in zip_longest(*[reversed(str(sv)) for sv in svs], fillvalue=0):
                res = str(sum([int(c) for c in cs])) + res
            return int(res)
        elif err == "forgetcarry":
            res = ""
            for cs in zip_longest(*[reversed(str(sv)) for sv in svs], fillvalue=0):
                res = str(sum([int(c) for c in cs]))[-1] + res
            return int(res)
        else:
            raise ValueError(f"Error type {err} not supported for addition")

    def _grad(self, wrt_vars: List[Any], instantiation):
        return sum([s.grad(wrt_vars, instantiation) for s in self.summands])

    def to_str(self, instantiation, depth: int, with_parentheses: bool = True) -> str:
        if depth > 0:
            summands_str = [a.to_str(instantiation, depth=depth-1, with_parentheses=with_parentheses) for a in self.summands]
            if with_parentheses:
                return " + ".join([f"({a})" for a in summands_str])
            else:
                return " + ".join([f"{a}" for a in summands_str])
        else:
            return str(self.eval(instantiation))
    
    def __str__(self) -> str:
        return " + ".join([f"({str(a)})" for a in self.summands])

class AdditionErr(SumErr):
    def __init__(self, summand1: Expr, summand2: Expr, mental_model: MentalModel) -> None:
        super().__init__(summands=[summand1, summand2], mental_model=mental_model)

class SubtractionErr(Expr):
    minuend: Expr
    subtrahend: Expr
    mental_model: MentalModel

    def __init__(self, minuend: Expr, subtrahend: Expr, mental_model: MentalModel) -> None:
        super().__init__(subexpressions=[minuend, subtrahend], minuend=minuend, subtrahend=subtrahend, mental_model=mental_model)

    def _eval(self, instantiation):
        m = self.minuend.eval(instantiation)
        s = self.subtrahend.eval(instantiation)

        err = self.mental_model.subtraction

        if err is None or (m < 10 and s < 10):
            # we require at least one 2-digit number for this to work
            return m - s
        elif err == "smallerfromlarge":
            res = ""
            for c1,c2 in zip_longest(reversed(str(m)),reversed(str(s)),fillvalue=0):
                res = str(abs(int(c1) - int(c2))) + res
            return int(res)
        else:
            raise ValueError(f"Error type {err} not supported for subtraction")

    def _grad(self, wrt_vars: List[Any], instantiation):
        # use normal math grad as proxy
        return self.minuend.grad(wrt_vars, instantiation) - self.subtrahend.grad(wrt_vars, instantiation)

    def to_str(self, instantiation, depth: int, with_parentheses: bool = True) -> str:
        if depth > 0:
            minuend_str = self.minuend.to_str(instantiation, depth=depth-1, with_parentheses=with_parentheses)
            subtrahend_str = self.subtrahend.to_str(instantiation, depth=depth-1, with_parentheses=with_parentheses)
            if with_parentheses:
                return f"({minuend_str}) - ({subtrahend_str})"
            else:
                return f"{minuend_str} - {subtrahend_str}"
        else:
            return str(self.eval(instantiation))

    def __str__(self) -> str:
        return f"({str(self.minuend)}) - ({str(self.subtrahend)})"
    
