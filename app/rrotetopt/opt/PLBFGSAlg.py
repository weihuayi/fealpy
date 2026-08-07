from collections import deque
from typing import Union, Deque

from fealpy.sparse import SparseTensor
from fealpy.operator import LinearOperator
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from .optimizer_base import Optimizer, Problem, Float, ObjFunc
from .line_search import wolfe_line_search

MatrixLike = Union[LinearOperator, SparseTensor, TensorLike, None]

class PLBFGS(Optimizer):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)

        self.S: Deque[Float] = deque(maxlen=self.problem.NumGrad)
        self.Y: Deque[Float] = deque(maxlen=self.problem.NumGrad)

        self._rho = bm.zeros(self.problem.NumGrad, dtype=bm.float64)
        self._alpha = bm.zeros(self.problem.NumGrad, dtype=bm.float64)

        # 统一预条件接口
        self.precond = getattr(problem, "Preconditioner", None)

    @classmethod
    def get_options(
        cls, *,
        x0: TensorLike,
        objective: ObjFunc,
        Preconditioner=None,
        MaxIters: int = 500,
        StepLengthTol: float = 1e-6,
        NormGradTol: float = 1e-6,
        NumGrad=10,
    ) -> Problem:
        return Problem(
            x0=x0,
            objective=objective,
            Preconditioner=Preconditioner,
            MaxIters=MaxIters,
            StepLengthTol=StepLengthTol,
            NormGradTol=NormGradTol,
            NumGrad=NumGrad
        )

    def _default_gamma(self, N: int) -> float:
        if N == 0:
            return 1.0

        s_last = self.S[-1]
        y_last = self.Y[-1]
        yty = float(y_last @ y_last)
        if yty > 0:
            return float((s_last @ y_last) / yty)
        return 1.0

    def hessian_gradient_prod(self, g: TensorLike) -> TensorLike:
        N = len(self.S)
        q = g.copy()

        rho = self._rho[:N]
        alpha = self._alpha[:N]

        for i in range(N - 1, -1, -1):
            sy = float(self.S[i] @ self.Y[i])
            rho[i] = 1.0 / sy
            alpha[i] = rho[i] * float(self.S[i] @ q)
            q -= alpha[i] * self.Y[i]

        # 初始逆Hessian作用
        if self.precond is None:
            r = self._default_gamma(N) * q
        else:
            r = self.precond.apply(q)
            if N > 0:
                gamma = self.precond.scale(self.S[-1], self.Y[-1])
                r *= gamma

        for i in range(N):
            beta = rho[i] * float(self.Y[i] @ r)
            r += (alpha[i] - beta) * self.S[i]

        return r

    def run(self):
        x = self.problem.x0
        f, g = self.fun(x)
        gnorm = bm.linalg.norm(g)

        if self.problem.Print:
            print(f'initial: nfval = {self.NF}, f = {f}, gnorm = {gnorm}')

        alpha = getattr(self.problem, "StepLength", 1.0)
        flag = 0
        j = 0

        if self.precond is not None:
            self.precond.setup(x)

        for k in range(1, self.problem.MaxIters + 1):
            d = -self.hessian_gradient_prod(g)
            gtd = float(g @ d)

            if gtd >= 0 or bm.isnan(gtd):
                print(f'Not descent direction, quit at iteration {k} with state f={f}, gnorm={gnorm}')
                break

            pg = g.copy()

            alpha, xalpha, falpha, galpha = wolfe_line_search(
                x, f, gtd, d, self.fun, alpha,
                project=self.problem.project_to_boundary
            )

            s = xalpha - x
            y = galpha - g
            sty = float(s @ y)

            eps_s = 1e-12
            stol = eps_s * (1.0 + bm.linalg.norm(x))
            if bm.linalg.norm(s) < stol:
                print(f'bfgs: norm(s) too small, restart BFGS memory at iteration {k}.')
                self.S.clear()
                self.Y.clear()
                alpha = 1.0
                x, f, g = xalpha, falpha, galpha
                gnorm = bm.linalg.norm(g)
                if self.precond is not None:
                    self.precond.update(x, k)
                continue

            if self.precond is not None:
                self.precond.update(xalpha, k)

            if self.problem.Print:
                print(f'current step {k}, StepLength = {alpha}, nfval = {self.NF}, f = {falpha}, gnorm = {bm.linalg.norm(galpha)}')

            if bm.abs(falpha - f) < self.problem.FunValDiff:
                x, f, g = xalpha, falpha, galpha
                flag = 1
                print(f"Convergence achieved after {k} iterations: |f_k+1-f_k| < FunValDiff,nfval={self.NF}")
                break

            if alpha <= self.problem.StepLengthTol:
                if j == 0:
                    flag = 2
                    print(f"Quit at iteration {k}: step length {alpha} <= StepLengthTol")
                    break
                else:
                    alpha = 1.0
                    self.S.clear()
                    self.Y.clear()
                    j = 0
                    x, f, g = xalpha, falpha, galpha
                    gnorm = bm.linalg.norm(g)
                    continue

            x, f, g = xalpha, falpha, galpha
            gnorm = bm.linalg.norm(g)

            if gnorm < self.problem.NormGradTol:
                flag = 1
                print(f"Convergence achieved after {k} iterations: ||g|| < NormGradTol")
                break

            if sty <= 0:
                print(f'bfgs: sty <= 0, skipping BFGS update at iteration {k}.')
            else:
                self.S.append(s)
                self.Y.append(y)
                j += 1

        if flag == 0:
            flag = 3

        return x, f, g
