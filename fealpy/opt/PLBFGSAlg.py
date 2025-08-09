from collections import deque
from typing import Union, Deque

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike 
from ..sparse import SparseTensor

from scipy.sparse.linalg import LinearOperator

from .optimizer_base import Optimizer, opt_alg_options
from .line_search_rules import StrongWolfeLineSearch

# 表示变量可以是 LinearOperator、稀疏矩阵或密集矩阵
MatrixLike = Union[LinearOperator, SparseTensor, TensorLike, None]

class PLBFGS(Optimizer):
    def __init__(self, options) -> None:
        super().__init__(options)

        self.S: Deque[bm.float64] = deque()
        self.Y: Deque[bm.float64] = deque()
        self.P = options["Preconditioner"]

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

    def hessian_gradient_prod(self, g: TensorLike) -> TensorLike:
        N = len(self.S)
        q = g
        rho = bm.zeros((N, ), dtype=bm.float64)
        alpha = bm.zeros((N, ), dtype=bm.float64)
        for i in range(N-1, -1, -1):
            rho = bm.set_at(rho, i, 1/bm.dot(self.S[i], self.Y[i]))
            alpha = bm.set_at(alpha, i, bm.dot(self.S[i], q)*rho[i])
            q = q - alpha[i]*self.Y[i]

        if self.P is not None:
            r = self.P@q
        else:
            r = q

        for i in range(0, N):
            beta = rho[i] * (bm.dot(self.Y[i], r))
            r = r + (alpha[i] - beta)*self.S[i]

        return r


    def run(self):
        options = self.options
        x = options["x0"]
        strongwolfe = StrongWolfeLineSearch()

        f, g = self.fun(x)
        gnorm = bm.linalg.norm(g)
        pg = g

        alpha = options["StepLength"]
        if options["Print"]:
            print('initial: f = {f}, gnorm = {gnorm}')

        flag = 0 # The convergence flag
        j = 0
        for i in range(1, options["MaxIters"]):
            d = -self.hessian_gradient_prod(g)
            gtd = bm.dot(g, d)

            if gtd >= 0 or bm.isnan(gtd):
                print(f'Not descent direction, quit at iteration {i} witht statt {f}, grad:{gnorm}')
                break

            pg = g

            alpha, xalpha, falpha, galpha = strongwolfe.search(x, f, gtd, d, self.fun, alpha)
            diff = bm.abs(falpha-f)
            x = xalpha
            f = falpha
            g = galpha
            gnorm = bm.linalg.norm(g)

            if options["Print"]:
                print(f'current step {i}, StepLength = {alpha}, ', end='')
                print(f'nfval = {self.NF}, f = {f}, gnorm = {gnorm}')

            if diff < options["FunValDiff"]:
                print(f"Convergence achieved after {i} iterations, the function value difference is less than FunValDiff")
                flag = 1
                break

            if gnorm < options["NormGradTol"]:
                print(f"The norm of current gradient is {gnorm}, which is smaller than the tolerance {options['NormGradTol']}")
                flag = 1
                break

            if alpha <= options["StepLengthTol"]:
                if j == 0:
                    flag = 2
                    break
                else:
                    alpha = 1
                    ND = x.shape[0]
                    self.S = deque()
                    self.Y = deque()
                    j = 0
                    continue
                        
            s = alpha*d
            y = g - pg
            sty = bm.dot(s, y)

            if sty < 0:
                print(f'bfgs: sty <= 0, skipping BFGS update at iteration {i}.')
            else:
                if i < options["NumGrad"]:
                    self.S.append(s)
                    self.Y.append(y)
                    j += 1
                else:
                    self.S.popleft()
                    self.S.append(s)
                    self.Y.popleft()
                    self.Y.append(y)

        if flag == 0:
            flag = 3
        return x, f, g, flag
