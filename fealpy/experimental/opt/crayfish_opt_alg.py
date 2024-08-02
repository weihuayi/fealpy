
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .optimizer_base import Optimizer, Problem


class CrayfishOptAlg(Optimizer):

    def __init__(self, problem: Problem) -> None:
        super().__init__(problme)

    @classmethod
    def get_options(
            cls, *,
            x0: TensorLike
            objective: ,
            ) -> Problem:
        return Problem(
                x0=x0,
                objective=objective,
                )

    def run(self):
        pass
