
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer, Problem


class SnowmeltOptAlg(Optimizer):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)

    @classmethod
    def get_options(
            cls,
            x0: TensorLike,
            objective,
            NP: int,
            MaxIters: int = 1000,
            MaxFunEvals: int = 10000,
            Print: bool = True,
            ) -> Problem:
        return Problem(
                x0,
                objective,
                NP=NP,
                MaxIters= MaxIters,
                MaxFunEvals=MaxFunEvals,
                Print=Print,
                )

    def run(self):
        pass

