from typing import TypedDict, Callable, Tuple, Union, Optional
import matplotlib.pyplot as plt
from fealpy.backend import backend_manager as bm 
from fealpy.typing import TensorLike, Index, _S
from fealpy import logger
from fealpy.opt.line_search_rules import *

def opt_alg_options(
    x0: TensorLike,
    objective,
    domain = None,
    NP: int = 1, # the number of solution points
    Preconditioner = None,
    MaxIters: int = 1000,
    MaxFunEvals: int = 10000,
    NormGradTol: float = 1e-6,
    FunValDiff: float = 1e-6,
    StepLength: float = 1.0,
    StepLengthTol: float = 1e-8,
    NumGrad: int = 10,
    LineSearch: Optional[LineSearch] = None,  # 默认值为 None,
    Print: bool = True,
):
    
    options = {
            "x0": x0,
            "objective": objective,
            "NP": NP,
            "ndim": x0.shape[-1],
            "domain": domain,
            "Preconditioner": Preconditioner,
            "MaxIters": MaxIters,
            "MaxFunEvals": MaxFunEvals,
            "NormGradTol": NormGradTol,
            "FunValDiff": FunValDiff,
            "StepLength": StepLength,
            "StepLengthTol": StepLengthTol,
            "NumGrad": NumGrad,
            "LineSearch": LineSearch,
            "Print": Print,
            }
    return options 

class Optimizer():
    def __init__(self, options) -> None:
        self.options = options 
        self.debug: bool = False
        self.__NF: int = 0
        self.x = options["x0"]
        self.N = options["NP"]
        self.MaxIT = options["MaxIters"]
        self.dim = options["ndim"]
        self.lb, self.ub = options["domain"]
        self.curve = bm.zeros((self.MaxIT,))
        self.D_pl = bm.zeros((self.MaxIT,))
        self.D_pt = bm.zeros((self.MaxIT,))
        self.Div = bm.zeros((1, self.MaxIT))

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
        self.__NF += self.options['NP'] 
        return self.options['objective'](x)

    def update_gbest(self, x, f):
        gbest_idx = bm.argmin(f)
        (self.gbest_f, self.gbest) = (f[gbest_idx], bm.copy(x[gbest_idx])) if f[gbest_idx] < self.gbest_f else (self.gbest_f, self.gbest)

    def run(self):
        raise NotImplementedError
    
    def D_pl_pt(self, it):
        self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(self.x, axis=0) - self.x)) / self.N)
        self.D_pl[it] = 100 * self.Div[0, it] / bm.max(self.Div)
        self.D_pt[it] = 100 * bm.abs(self.Div[0, it] - bm.max(self.Div)) / bm.max(self.Div)

    def print_optimal_result(self):
        print(f"Optimal solution: {self.gbest} \nFitness: {self.gbest_f}")

    def plot_plpt_percen(self):
        plt.plot(self.D_pl, label=(f'Exploration:{bm.round(bm.mean(self.D_pl), 2)}%'))
        plt.plot(self.D_pt, label=(f'Exploitation:{bm.round(bm.mean(self.D_pt), 2)}%'))
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Percentage')
        plt.title("exploration vs exploitation percentage")
        plt.show()  

    def plot_curve(self, label="label"):
        plt.semilogy(self.curve, label=label)
        plt.xlabel('Iteration')
        plt.ylabel("Best fitness obtained so far")
        plt.title("Convergence curve")
        plt.legend()
        plt.show()