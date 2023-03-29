
from typing import TypedDict, Callable, Tuple, Union
from numpy.typing import NDArray
import numpy as np


ObjFunc = Callable[
    [NDArray],
    Tuple[np.floating, NDArray[np.floating]]
]

Float = Union[float, np.floating]


class Problem(TypedDict):
    """
    @brief Problem dict for optimizers.

    @param x0: NDArray. Initial point for algorithms.
    @param objective: Callable. The objective function which should return\
                      it's value and gradient as a tuple.
    """
    x0: NDArray
    objective: ObjFunc


class Options(TypedDict, total=False):
    """
    @brief Options for optimizers.

    @param MaxIters: int.
    @param MaxFunEvals: int.
    @param NormGradTol: float.
    @param FunValDiff: float.
    @param StepLength: float.
    @param StepTol: float.
    @param Disp: bool.
    @param Output: bool.
    @param Preconditioner: NDArray.
    """

    MaxIters: int
    MaxFunEvals: int
    NormGradTol: float
    FunValDiff: float
    StepLength: float
    StepTol: float
    Disp: bool
    Output: bool
    Preconditioner: NDArray


options = Options()

# TODO: Finish this


class Optimizer():
    def __init__(self, problem: Problem, options: Options) -> None:
        self.problem = problem
        self.options = options

        self.debug: bool = False
        self.__NF: int = 0


    @classmethod
    def get_options(cls, *args, **kwargs):
        options = Options()
        options.update(kwargs)
        return options


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
        return self.problem['objective'](x)


    def run(self):
        raise NotImplementedError
