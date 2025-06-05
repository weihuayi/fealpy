from collections import deque
from typing import Union, Deque

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from ..sparse import SparseTensor
from scipy.sparse.linalg import LinearOperator
from .optimizer_base import Optimizer,opt_alg_options
from .line_search_rules import StrongWolfeLineSearch

# 表示变量可以是 LinearOperator、稀疏矩阵或密集矩阵
MatrixLike = Union[LinearOperator, SparseTensor, TensorLike, None]

class PNLCG(Optimizer):
    def __init__(self, options) -> None:
        super().__init__(options)
        self.P: MatrixLike = options["Preconditioner"]

    @classmethod
    def get_options(
        cls, *,
        x0: TensorLike,
        objective,
        Preconditioner: MatrixLike = None,
        MaxIters: int = 500,
        StepLengthTol: float = 1e-6,
        NormGradTol: float = 1e-6,
        NumGrad = 10,
    ):

        return opt_alg_options(
            x0=x0,
            objective=objective,
            Preconditioner=Preconditioner,
            MaxIters=MaxIters,
            StepLengthTol=StepLengthTol,
            NormGradTol=NormGradTol,
            NumGrad=NumGrad
        )
    def scalar_coefficient(self,g0,g1,stype='PR'):
        if stype =='PR':
            beta = bm.dot(g1,g1-g0)/bm.dot(g0,g0)
            beta = max(0.0,beta)
        return beta

    def run(self,stype='PR'):
        options = self.options
        x = options["x0"]
        strongwolfe = StrongWolfeLineSearch()
        flag = True

        f, g = self.fun(x)
        gnorm = bm.linalg.norm(g)
        if options["Print"]:
            print(f'initial:  f = {f}, gnorm = {gnorm}')
        alpha = 1

        if self.P is None:
            d = -g
        else:
            pg0 = self.P@g
            d = -pg0

        for i in range(1, options["MaxIters"]+1):
            gtd = bm.dot(g, d)
             
            if gtd >= 0 or bm.isnan(gtd):
                print(f'Not descent direction quit at iteration {i} witht statt {f}, grad:{gnorm}')
                break
            
            alpha, xalpha, falpha, galpha = strongwolfe.search(x, f, gtd, d, self.fun, alpha)
            gnorm = bm.linalg.norm(g)
            if options["Print"]:
                print(f'current step {i}, StepLength = {alpha}, ', end='')
                print(f'nfval = {self.NF}, f = {falpha}, gnorm = {gnorm}')

            if bm.abs(falpha - f) < options["FunValDiff"]:
                print(f"Convergence achieved after {i} iterations, the function value difference is less than FunValDiff")
                x = xalpha
                f = falpha
                g = galpha
                break
            
            if gnorm < options["NormGradTol"]:
                print(f"The norm of current gradient is {gnorm}, which is smaller than the tolerance {self.problem.NormGradTol}")
                x = xalpha
                f = falpha
                g = galpha
                break
            
            if alpha < options["StepLengthTol"]:
                print(f"The step length is smaller than the tolerance {self.problem.StepLengthTol}")
                x = xalpha
                f = falpha
                g = galpha
                break

            x = xalpha
            f = falpha 
            if self.P is None:
                beta = self.scalar_coefficient(g,galpha,stype=stype)
                g = galpha
                d = -g + beta*d
            else:
                pg1 = self.P@galpha
                beta = self.scalar_coefficient(pg0,pg1,stype=stype)
                g = galpha
                pg0 = pg1
                d = -pg0 + beta*d

        return x, f, g
