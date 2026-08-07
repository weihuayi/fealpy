from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .optimizer_base import Optimizer, Problem, Float, ObjFunc
from .line_search import wolfe_line_search

class PNLCG(Optimizer):
    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
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
    ) -> Problem:
        return Problem(
            x0=x0,
            objective=objective,
            Preconditioner=Preconditioner,
            MaxIters=MaxIters,
            StepLengthTol=StepLengthTol,
            NormGradTol=NormGradTol,
        )

    @staticmethod
    def _beta_pr(g0: TensorLike, g1: TensorLike, eps=1e-30) -> float:
        den = float(g0 @ g0)
        if not bm.isfinite(den) or den <= eps:
            return 0.0
        num = float(g1 @ (g1 - g0))
        if not bm.isfinite(num):
            return 0.0
        beta = num / den
        if not bm.isfinite(beta) or beta < 0.0:
            beta = 0.0
        return beta

    @staticmethod
    def _beta_pr_precond(g0: TensorLike, z0: TensorLike, g1: TensorLike, z1: TensorLike, eps=1e-30) -> float:
        den = float(g0 @ z0)
        if not bm.isfinite(den) or den <= eps:
            return 0.0
        num = float(g1 @ (z1 - z0))
        if not bm.isfinite(num):
            return 0.0
        beta = num / den
        if not bm.isfinite(beta) or beta < 0.0:
            beta = 0.0
        return beta

    def run(self):
        x = self.problem.x0
        f, g = self.fun(x)
        gnorm = bm.linalg.norm(g)

        print(f'initial: nfval = {self.NF}, f = {f}, gnorm = {gnorm}')
        alpha = 1.0

        if self.precond is None:
            z = None
            d = -g
        else:
            self.precond.setup(x)
            z = self.precond.apply(g)
            d = -z

        for i in range(1, self.problem.MaxIters + 1):
            gtd = float(g @ d)

            if gtd >= 0 or bm.isnan(gtd):
                if self.precond is not None:
                    z = self.precond.apply(g)
                    d = -z
                else:
                    d = -g
                gtd = float(g @ d)
                if gtd >= 0 or bm.isnan(gtd):
                    print(f'Not descent direction, quit at iteration {i} with state f={f}, gnorm={gnorm}')
                    break

            alpha, xalpha, falpha, galpha = wolfe_line_search(
                x, f, gtd, d, self.fun, alpha,
                project=self.problem.project_to_boundary
            )

            gnorm = bm.linalg.norm(galpha)
            if self.problem.Print:
                print(f'current step {i}, StepLength = {alpha}, nfval = {self.NF}, f = {falpha}, gnorm = {gnorm}')

            if bm.abs(falpha - f) < self.problem.FunValDiff:
                print(f"Convergence achieved after {i} iterations: |f_k+1-f_k| < FunValDiff,nfval={self.NF}")
                x, f, g = xalpha, falpha, galpha
                break

            if gnorm < self.problem.NormGradTol:
                print(f"The norm of current gradient is {gnorm}, which is smaller than tolerance")
                x, f, g = xalpha, falpha, galpha
                break

            if alpha < self.problem.StepLengthTol:
                print(f"The step length is smaller than the tolerance {self.problem.StepLengthTol}")
                x, f, g = xalpha, falpha, galpha
                break

            if self.precond is None:
                beta = self._beta_pr(g, galpha)
                d = -galpha + beta * d
                if float(galpha @ d) >= 0 or bm.isnan(float(galpha @ d)):
                    d = -galpha
                x, f, g = xalpha, falpha, galpha
            else:
                self.precond.update(xalpha, i)
                z1 = self.precond.apply(galpha)
                beta = self._beta_pr_precond(g, z, galpha, z1)
                d = -z1 + beta * d
                if float(galpha @ d) >= 0 or bm.isnan(float(galpha @ d)):
                    d = -z1
                x, f, g, z = xalpha, falpha, galpha, z1

        return x, f, g
