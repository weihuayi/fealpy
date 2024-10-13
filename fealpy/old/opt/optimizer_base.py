from typing import TypedDict, Callable, Tuple, Union, Optional

from scipy.sparse.linalg import LinearOperator
from scipy.sparse import spmatrix

from numpy.typing import NDArray

import numpy as np


ObjFunc = Callable[
    [NDArray],
    Tuple[np.floating, NDArray[np.floating]]
]

Float = Union[float, np.floating]
# 表示变量可以是 LinearOperator、稀疏矩阵或密集矩阵
MatrixLike = Union[LinearOperator, spmatrix, np.ndarray, None]

class Problem():
    def __init__(
        self,
        x0: NDArray,
        objective: ObjFunc,
        Preconditioner: MatrixLike = None,
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
        @brief The number of times the function value and gradient are calculated.
        """
        return self.__NF


    def fun(self, x: NDArray):
        """
        @brief Objective function.
        The counter `self.NF` works automatically when call `fun(x)`.

        @param x: NDArray. Input of objective function.

        @return: Function value and gradient.
        """
        self.__NF += 1
        return self.problem.objective(x)


    def run(self):
        raise NotImplementedError
