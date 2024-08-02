from typing import TypedDict, Callable, Tuple, Union, Optional

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger


class Problem():
    def __init__(
        self,
        x0: TensorLike,
        objjective: ,
        NN: int = 1,
        Preconditioner: = None,
        MaxIters: int = 1000,
        MaxFunEvals: int = 10000,
        NormGradTol: float = 1e-6,
        FunValDiff: float = 1e-6,
        StepLength: float = 1.0,
        StepLengthTol: float = 1e-8,
        NumGrad: int = 10,
        Print: bool = True,
        Linesearch: Optional[str] = None 
    ):
        self.x0 = x0
        self.objective = objective
        self.NN = NN # number of solution 
        self.Preconditioner = Preconditioner
        self.MaxIters = MaxIters
        self.MaxFunEvals = MaxFunEvals
        self.NormGradTol = NormGradTol
        self.FunValDiff = FunValDiff
        self.StepLength = StepLength
        self.StepLengthTol = StepLengthTol
        self.NumGrad = NumGrad
        self.Print = Print
        self.Linesearch = Linesearch

class Optimizer():
    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.debug: bool = False
        self.__NF: int = 0

    @property
    def NF(self) -> int:
        """
        The number of times the function value and gradient are calculated.
        """
        return self.__NF


    def fun(self, x: TensorLike):
        """
        Objective function.
        The counter `self.NF` works automatically when call `fun(x)`.

        Parameters:
            x [TensorLike]: Input of objective function.

        Return:
            The function value, with gradient value for gradient methods.
        """
        self.__NF += problem.NN 
        return self.problem.objective(x)


    def run(self):
        raise NotImplementedError
